from flask import Blueprint, request, jsonify, current_app, send_from_directory
from flask_jwt_extended import jwt_required, get_jwt_identity
from PIL import Image
import base64, os, io, threading
import numpy as np
import cv2

from extensions import db
from models import (
    ImageAsset, SegmentationResult,
    VideoAsset, VideoSegmentationResult,
)
from utils_storage import save_pil_image, save_upload_file, unique_name
from yolo_seg import (
    YoloSegmenter,
    mask_to_polygon_and_obb,
    apply_overlay,
    pngmask_to_b64,
    pil_to_bgr,
    bgr_to_b64jpg,
)

bp_vision = Blueprint("vision", __name__, url_prefix="/api/vision")

seg = YoloSegmenter()

def b64_to_nd(mask_b64: str) -> np.ndarray:
    arr = np.frombuffer(base64.b64decode(mask_b64.split(",")[-1]), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

def public_url(filename: str) -> str:
    return f"/storage/{filename}"

@bp_vision.get("/storage/<path:fname>")
def storage_public(fname: str):
    storage_dir = current_app.config["STORAGE_DIR"]
    return send_from_directory(storage_dir, fname, as_attachment=False)


@bp_vision.post("/segment/auto")
@jwt_required()
def segment_auto():
    if "file" not in request.files:
        return jsonify({"error": "sube imagen en 'file'"}), 400

    conf = float(request.args.get("conf", 0.25))
    imgsz = int(request.args.get("imgsz", 640))
    want_save = request.args.get("save", "0").lower() in ("1", "true", "yes")

    user_id = int(get_jwt_identity())
    storage_dir = current_app.config["STORAGE_DIR"]

    upload = request.files["file"]
    img = Image.open(upload.stream).convert("RGB")
    width, height = img.size

    saved_image = None
    if want_save:
        orig_name, _ = save_pil_image(img, storage_dir, ext=".jpg", quality=92)
        saved_image = ImageAsset(
            user_id=user_id,
            filename=orig_name,
            mime="image/jpeg",
            width=width,
            height=height,
        )
        db.session.add(saved_image)
        db.session.flush()

    out = seg.segment_pil(img, conf=conf, imgsz=imgsz)

    result_row = None
    overlay_url = out.get("overlay_jpg_b64")
    if want_save and overlay_url and saved_image:
        b64data = overlay_url.split(",")[-1]
        buf = base64.b64decode(b64data)
        overlay = Image.open(io.BytesIO(buf)).convert("RGB")

        ov_name, _ = save_pil_image(overlay, storage_dir, ext=".jpg", quality=90)
        result_row = SegmentationResult(
            image_id=saved_image.id,
            overlay_filename=ov_name,
            objects_json=out.get("objects", []),
        )
        db.session.add(result_row)
        db.session.flush()

        saved_image.last_result_id = result_row.id
        db.session.commit()

        out["saved"] = True
        out["image"] = {
            "id": saved_image.id,
            "original_url": public_url(saved_image.filename),
            "overlay_url": public_url(ov_name),
            "width": width,
            "height": height,
            "result_id": result_row.id,
        }
    else:
        out["saved"] = False

    try:
        H, W = pil_to_bgr(img).shape[:2]
    except Exception:
        H = W = None
    print(f"[segment/auto] user={user_id} detections={len(out.get('objects', []))} conf={conf} imgsz={imgsz} shape={(H,W)} saved={want_save}")

    return jsonify(out), 200

@bp_vision.get("/images")
@jwt_required()
def list_images():
    user_id = int(get_jwt_identity())
    rows = (
        ImageAsset.query
        .filter_by(user_id=user_id)
        .order_by(ImageAsset.created_at.desc())
        .all()
    )
    data = []
    for r in rows:
        item = {
            "id": r.id,
            "original_url": public_url(r.filename),
            "width": r.width,
            "height": r.height,
            "created_at": r.created_at.isoformat(),
        }
        if r.last_result:
            item["overlay_url"] = public_url(r.last_result.overlay_filename)
            item["last_result_id"] = r.last_result.id
        data.append(item)
    return jsonify(data), 200

@bp_vision.get("/images/<int:image_id>")
@jwt_required()
def image_detail(image_id: int):
    user_id = int(get_jwt_identity())
    r = ImageAsset.query.filter_by(id=image_id, user_id=user_id).first_or_404()
    results = (
        SegmentationResult.query
        .filter_by(image_id=r.id)
        .order_by(SegmentationResult.created_at.desc())
        .all()
    )
    return jsonify({
        "id": r.id,
        "original_url": public_url(r.filename),
        "width": r.width,
        "height": r.height,
        "results": [{
            "id": res.id,
            "overlay_url": public_url(res.overlay_filename),
            "objects": res.objects_json,
            "created_at": res.created_at.isoformat(),
        } for res in results]
    }), 200


def _transcode_to_mp4(src_path: str) -> str | None:
    try:
        root, ext = os.path.splitext(src_path)
        if ext.lower() != ".avi":
            return None
        dst_path = root + ".mp4"
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-i", src_path,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            dst_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try: os.remove(src_path)
        except: pass
        return dst_path
    except Exception as e:
        print(f"[segment/video] ⚠️ ffmpeg falló: {e}")
        return None


def _process_video_async(app_ctx, va_id, orig_path, out_path, conf, imgsz, sample):
    global seg
    from models import VideoAsset, VideoSegmentationResult

    with app_ctx:
        try:
            print("[segment/video] D) inferencia YOLO...")
            info = seg.segment_video_path(
                in_path=orig_path,
                out_path=out_path,
                conf=conf,
                imgsz=imgsz,
                sample_every=sample,
                max_side=1280
            )
            final_path = info["out_path"]
            final_name = os.path.basename(final_path)
            mp4_path = _transcode_to_mp4(final_path)
            if mp4_path and os.path.exists(mp4_path):
                final_path = mp4_path
                final_name = os.path.basename(mp4_path)
                print(f"[segment/video] 🎬 transcodificado a MP4: {final_name}")

            print("[segment/video] E) guardando resultado en DB...")
            vres = VideoSegmentationResult(
                video_id=va_id,
                overlay_filename=final_name,
                objects_json={
                    "totals": info["objects_totals"],
                    "conf": conf,
                    "imgsz": imgsz,
                    "sample_every": sample
                },
            )
            db.session.add(vres)
            db.session.flush()

            va = db.session.get(VideoAsset, va_id)
            if va:
                va.last_result_id = vres.id

            db.session.commit()
            print(f"[segment/video] ✅ listo {final_name} (vres_id={vres.id})")

        except Exception as e:
            db.session.rollback()
            print(f"[segment/video] 💥 error: {e}")

        finally:
            db.session.remove()

@bp_vision.post("/segment/video")
@jwt_required()
def segment_video():
    print("[segment/video] ▶️ request recibido")
    if "file" not in request.files:
        return jsonify({"error": "sube video en 'file'"}), 400

    conf   = float(request.args.get("conf", 0.25))
    imgsz  = int(request.args.get("imgsz", 640))
    sample = max(int(request.args.get("sample", 1)), 1)

    user_id = int(get_jwt_identity())
    storage_dir = current_app.config["STORAGE_DIR"]
    os.makedirs(storage_dir, exist_ok=True)

    up = request.files["file"]

    print("[segment/video] A) guardando upload...")
    orig_name, orig_path = save_upload_file(up, storage_dir, fallback_ext=".mp4")
    print(f"[segment/video] 💾 {orig_name} -> {orig_path}")

    print("[segment/video] B) leyendo metadatos...")
    import cv2 as _cv2
    cap = _cv2.VideoCapture(orig_path)
    if not cap.isOpened():
        return jsonify({"error": "No se pudo abrir el video"}), 400
    fps = cap.get(_cv2.CAP_PROP_FPS) or 25.0
    W  = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
    N  = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    duration = float(N) / float(fps) if fps > 0 else 0.0
    cap.release()

    if duration > 60.0:
        try: os.remove(orig_path)
        except: pass
        return jsonify({"error": "El video excede 60s"}), 400

    print("[segment/video] C) insert VideoAsset...")
    va = VideoAsset(
        user_id=user_id,
        filename=orig_name,
        mime=up.mimetype or "video/mp4",
        width=W, height=H, fps=float(fps), duration_sec=float(duration),
    )
    db.session.add(va)
    db.session.flush()
    db.session.commit()
    print(f"[segment/video] C’) committed va.id={va.id}")

    out_name = unique_name(".avi")
    out_path = os.path.join(storage_dir, out_name)

    threading.Thread(
        target=_process_video_async,
        args=(current_app.app_context(), va.id, orig_path, out_path, conf, imgsz, sample),
        daemon=True
    ).start()

    return jsonify({
        "accepted": True,
        "video": {
            "id": va.id,
            "original_url": public_url(va.filename),
            "width": W, "height": H,
            "fps": float(fps), "duration_sec": float(duration),
        },
        "message": "Procesando; consulta /api/vision/videos o /api/vision/videos/<id> para ver overlay_url cuando esté listo"
    }), 202

@bp_vision.get("/videos")
@jwt_required()
def videos_list():
    user_id = int(get_jwt_identity())
    rows = (
        VideoAsset.query
        .filter_by(user_id=user_id)
        .order_by(VideoAsset.created_at.desc())
        .all()
    )
    out = []
    for r in rows:
        item = {
            "id": r.id,
            "original_url": public_url(r.filename),
            "width": r.width,
            "height": r.height,
            "fps": r.fps,
            "duration_sec": r.duration_sec,
            "created_at": r.created_at.isoformat(),
        }
        if r.last_result:
            item["overlay_url"] = public_url(r.last_result.overlay_filename)
            item["last_result_id"] = r.last_result.id
            item["objects_totals"] = (r.last_result.objects_json or {}).get("totals", {})
        out.append(item)
    return jsonify(out), 200

@bp_vision.get("/videos/<int:video_id>")
@jwt_required()
def video_detail(video_id: int):
    user_id = int(get_jwt_identity())
    v = VideoAsset.query.filter_by(id=video_id, user_id=user_id).first_or_404()
    results = (
        VideoSegmentationResult.query
        .filter_by(video_id=v.id)
        .order_by(VideoSegmentationResult.created_at.desc())
        .all()
    )
    return jsonify({
        "id": v.id,
        "original_url": public_url(v.filename),
        "width": v.width,
        "height": v.height,
        "fps": v.fps,
        "duration_sec": v.duration_sec,
        "created_at": v.created_at.isoformat(),
        "results": [{
            "id": r.id,
            "overlay_url": public_url(r.overlay_filename),
            "objects_totals": (r.objects_json or {}).get("totals", {}),
            "meta": r.objects_json,
            "created_at": r.created_at.isoformat(),
        } for r in results]
    }), 200
