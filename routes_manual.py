from flask import Blueprint, request, jsonify, current_app, send_from_directory
from flask_jwt_extended import jwt_required, get_jwt_identity
from extensions import db
from models import EtiquetaManualAsset, EtiquetaManualResult
from utils_storage import save_pil_image, save_upload_file, unique_name
from PIL import Image
import os, io, base64
import os
from PIL import Image, ImageDraw
from PIL import ImageFont
font = ImageFont.load_default()

bp_manual = Blueprint("manual", __name__, url_prefix="/api/manual")

def public_url(filename: str):
    base = os.getenv("PUBLIC_BACKEND_URL", "http://localhost:5051")
    return f"{base}/storage/{filename}"
# =====================
# 1. SUBIR IMAGEN
# =====================
@bp_manual.post("/upload")
@jwt_required()
def upload_manual_image():
    if "file" not in request.files:
        return jsonify({"error": "sube imagen en 'file'"}), 400

    user_id = int(get_jwt_identity())
    storage_dir = current_app.config["STORAGE_DIR"]

    up = request.files["file"]
    img = Image.open(up.stream).convert("RGB")
    width, height = img.size

    # Guardar imagen original
    filename, path = save_pil_image(img, storage_dir, ext=".jpg", quality=92)

    row = EtiquetaManualAsset(
        user_id=user_id,
        filename=filename,
        width=width,
        height=height
    )
    db.session.add(row)
    db.session.commit()

    return jsonify({
        "id": row.id,
        "url": public_url(filename),
        "width": width,
        "height": height
    }), 201


# =====================
# 2. LISTAR IMÁGENES
# =====================
@bp_manual.get("/images")
@jwt_required()
def manual_images_list():
    user_id = int(get_jwt_identity())
    rows = (
        EtiquetaManualAsset
        .query
        .filter_by(user_id=user_id)
        .order_by(EtiquetaManualAsset.created_at.desc())
        .all()
    )

    out = []
    for r in rows:
        out.append({
            "id": r.id,
            "url": public_url(r.filename),
            "width": r.width,
            "height": r.height,
            "created_at": r.created_at.isoformat()
        })

    return jsonify(out), 200


# =====================
# 3. DETALLE + RESULTADOS
# =====================
@bp_manual.get("/images/<int:asset_id>")
@jwt_required()
def manual_image_detail(asset_id):
    user_id = int(get_jwt_identity())

    asset = (
        EtiquetaManualAsset
        .query
        .filter_by(id=asset_id, user_id=user_id)
        .first_or_404()
    )

    results = (
        EtiquetaManualResult.query
        .filter_by(asset_id=asset.id)
        .order_by(EtiquetaManualResult.created_at.desc())
        .all()
    )

    return jsonify({
        "id": asset.id,
        "url": public_url(asset.filename),
        "width": asset.width,
        "height": asset.height,
        "results": [{
            "id": res.id,
            "annotations": res.annotations,
            "created_at": res.created_at.isoformat()
        } for res in results]
    }), 200


# =====================
# 4. GUARDAR ANOTACIONES
# =====================
@bp_manual.post("/images/<int:asset_id>/save")
@jwt_required()
def save_annotations(asset_id):
    user_id = int(get_jwt_identity())

    asset = (
        EtiquetaManualAsset
        .query
        .filter_by(id=asset_id, user_id=user_id)
        .first_or_404()
    )

    data = request.get_json(force=True) or {}
    ann = data.get("annotations")

    if not isinstance(ann, list):
        return jsonify({"error": "annotations debe ser lista"}), 400

    # === 1) Cargar imagen original ===
    storage_dir = current_app.config["STORAGE_DIR"]
    img_path = os.path.join(storage_dir, asset.filename)

    # → Importante: usar RGBA si quieres relleno
    img = Image.open(img_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    # === 2) Dibujar TODAS las anotaciones ===
    font = ImageFont.load_default()

    for a in ann:
        t = a.get("type")
        cls = a.get("class", "")

        # ---- BBOX ----
        if t == "bbox":
            x1, y1, x2, y2 = a["bbox"]

            # normalizar
            x0 = min(x1, x2)
            x1 = max(x1, x2)
            y0 = min(y1, y2)
            y1 = max(y1, y2)

            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)

            # texto encima
            label = f"{cls} ({t})"
            draw.text(
                (x0 + 5, y0 - 15),
                label,
                fill="white",
                font=font,
                stroke_width=3,
                stroke_fill="black"
            )

        # ---- OBB ----
        elif t == "obb":
            pts = a.get("points", [])
            if pts:
                draw.polygon(pts, outline="orange", width=3)

                # texto en el primer punto
                x, y = pts[0]
                label = f"{cls} ({t})"
                draw.text(
                    (x + 5, y - 15),
                    label,
                    fill="white",
                    font=font,
                    stroke_width=3,
                    stroke_fill="black"
                )

                # ---- SEG ----
        elif t == "seg":
            pts = a.get("points", [])
            if pts:
                # --- CAPA TRANSPARENTE ---
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                draw_ov = ImageDraw.Draw(overlay)
                fill_color = (0, 255, 255, 80)      # transparencia real
                outline_color = (0, 180, 255, 255)  # contorno visible
                # DIBUJAR EN LA CAPA
                draw_ov.polygon(pts, fill=fill_color, outline=outline_color)

                # MERGE con imagen
                img = Image.alpha_composite(img, overlay)
                draw = ImageDraw.Draw(img)   # actualizar draw base
                # TEXTO con borde
                x, y = pts[0]
                label = f"{cls} ({t})"
                draw.text(
                        (x + 5, y - 15),
                        label,
                        fill="white",
                        font=font,
                        stroke_width=3,
                        stroke_fill="black"
                )

        # ---- POSE ----
        elif t == "pose":
            pts = a.get("points", [])
            for (px, py) in pts:
                draw.ellipse([px-3, py-3, px+3, py+3], fill="yellow")

            if pts:
                x, y = pts[0]
                label = f"{cls} ({t})"
                draw.text(
                    (x + 5, y - 15),
                    label,
                    fill="white",
                    font=font,
                    stroke_width=3,
                    stroke_fill="black"
                )



    # === 3) Convertir a RGB (JPEG no soporta RGBA) ===
    final = img.convert("RGB")

    # === 4) Guardar overlay ===
    overlay_name = unique_name(".jpg")
    overlay_path = os.path.join(storage_dir, overlay_name)
    final.save(overlay_path, "JPEG", quality=90)

    # === 5) Guardar registro ===
    res = EtiquetaManualResult(
        asset_id=asset.id,
        annotations=ann,
        overlay_filename=overlay_name
    )

    db.session.add(res)
    db.session.commit()

    return jsonify({
        "saved": True,
        "result_id": res.id,
        "overlay_url": public_url(overlay_name)
    }), 201


# =====================
# 5. EXPORTAR FORMATOS
# =====================

# ---- JSON estilo LabelMe ----
@bp_manual.get("/images/<int:asset_id>/export/json")
@jwt_required()
def export_json(asset_id):
    user_id = int(get_jwt_identity())

    asset = (
        EtiquetaManualAsset
        .query
        .filter_by(id=asset_id, user_id=user_id)
        .first_or_404()
    )

    last = (
        EtiquetaManualResult
        .query
        .filter_by(asset_id=asset.id)
        .order_by(EtiquetaManualResult.created_at.desc())
        .first()
    )

    if not last:
        return jsonify({"error": "no hay anotaciones"}), 404

    return jsonify({
        "image": asset.filename,
        "width": asset.width,
        "height": asset.height,
        "annotations": last.annotations
    }), 200


# ---- YOLO style (solo bbox) ----
@bp_manual.get("/images/<int:asset_id>/export/yolo")
@jwt_required()
def export_yolo(asset_id):
    user_id = int(get_jwt_identity())

    asset = (
        EtiquetaManualAsset
        .query
        .filter_by(id=asset_id, user_id=user_id)
        .first_or_404()
    )

    last = (
        EtiquetaManualResult
        .query
        .filter_by(asset_id=asset.id)
        .order_by(EtiquetaManualResult.created_at.desc())
        .first()
    )

    if not last:
        return jsonify({"error": "no hay anotaciones"}), 404

    W = asset.width
    H = asset.height

    lines = []
    for ann in last.annotations:
        if ann.get("type") != "bbox":
            continue

        cls = ann.get("class", "obj")
        x1, y1, x2, y2 = ann["bbox"]

        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        ww = (x2 - x1) / W
        hh = (y2 - y1) / H

        lines.append(f"{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")

    return jsonify({
        "filename": asset.filename.replace(".jpg", ".txt"),
        "content": "\n".join(lines)
    }), 200
@bp_manual.get("/images/<int:asset_id>/overlay")
def manual_overlay(asset_id):
    from flask import send_from_directory, abort

    storage_dir = current_app.config["STORAGE_DIR"]

    asset = (
        EtiquetaManualAsset
        .query
        .filter_by(id=asset_id)
        .first()
    )
    if not asset:
        abort(404)

    last = (
        EtiquetaManualResult
        .query
        .filter_by(asset_id=asset.id)
        .order_by(EtiquetaManualResult.created_at.desc())
        .first()
    )

    if not last or not last.overlay_filename:
        abort(404)

    # DELIVERY directo del .jpg generado
    return send_from_directory(storage_dir, last.overlay_filename)

@bp_manual.post("/images/batch")
@jwt_required()
def manual_images_batch():
    data = request.get_json() or {}
    ids = data.get("ids", [])

    if not isinstance(ids, list):
        return jsonify({"error": "ids debe ser lista"}), 400

    user_id = int(get_jwt_identity())

    rows = (
        EtiquetaManualAsset.query
        .filter(EtiquetaManualAsset.user_id == user_id)
        .filter(EtiquetaManualAsset.id.in_(ids))
        .order_by(EtiquetaManualAsset.id.asc())
        .all()
    )

    out = []
    for r in rows:
        out.append({
            "id": r.id,
            "url": public_url(r.filename),
            "width": r.width,
            "height": r.height,
        })

    return jsonify(out), 200
@bp_manual.post("/batch/save")
@jwt_required()
def manual_batch_save():
    from PIL import Image, ImageDraw, ImageFont

    data = request.get_json() or {}
    items = data.get("items", [])

    if not isinstance(items, list):
        return jsonify({"error": "items debe ser lista"}), 400

    user_id = int(get_jwt_identity())

    # ============================
    # 🔥 VALIDACIÓN PREVIA GLOBAL
    # ============================
    for item in items:
        asset_id = item.get("id")
        ann = item.get("annotations")

        if not asset_id:
            return jsonify({"error": "Falta id en un item del batch"}), 400

        # No es lista → inválido
        if not isinstance(ann, list):
            return jsonify({
                "error": f"La imagen {asset_id} no tiene anotaciones válidas."
            }), 400

        # Lista vacía → no se permite batch
        if len(ann) == 0:
            return jsonify({
                "error": f"La imagen {asset_id} no tiene anotaciones. Debes completar todas antes de guardar."
            }), 400

    # Si pasamos la validación, guardamos TODO
    storage_dir = current_app.config["STORAGE_DIR"]
    saved = []

    for item in items:
        asset_id = item["id"]
        ann = item["annotations"]

        asset = (
            EtiquetaManualAsset
            .query
            .filter_by(id=asset_id, user_id=user_id)
            .first()
        )

        if not asset:
            return jsonify({"error": f"El asset {asset_id} no existe o no es tuyo."}), 404

        # === Cargar imagen original ===
        img_path = os.path.join(storage_dir, asset.filename)
        img = Image.open(img_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        # === Dibujar TODAS las anotaciones ===
        for a in ann:
            t = a.get("type")
            cls = a.get("class", "")

            if t == "bbox":
                x1, y1, x2, y2 = a["bbox"]
                x0 = min(x1, x2)
                x1 = max(x1, x2)
                y0 = min(y1, y2)
                y1 = max(y1, y2)

                draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                draw.text(
                    (x0 + 5, y0 - 15),
                    f"{cls} (bbox)",
                    fill="white",
                    font=font,
                    stroke_width=3,
                    stroke_fill="black"
                )

            elif t == "obb":
                pts = a.get("points", [])
                if pts:
                    draw.polygon(pts, outline="orange", width=3)
                    x, y = pts[0]
                    draw.text(
                        (x + 5, y - 15),
                        f"{cls} (obb)",
                        fill="white",
                        font=font,
                        stroke_width=3,
                        stroke_fill="black"
                    )

            elif t == "seg":
                pts = a.get("points", [])
                if pts:
                    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    draw_ov = ImageDraw.Draw(overlay)
                    fill_color = (0, 255, 255, 80)
                    outline_color = (0, 180, 255, 255)

                    draw_ov.polygon(pts, fill=fill_color, outline=outline_color)

                    img = Image.alpha_composite(img, overlay)
                    draw = ImageDraw.Draw(img)

                    x, y = pts[0]
                    draw.text(
                        (x + 5, y - 15),
                        f"{cls} (seg)",
                        fill="white",
                        font=font,
                        stroke_width=3,
                        stroke_fill="black"
                    )

            elif t == "pose":
                pts = a.get("points", [])
                for (px, py) in pts:
                    draw.ellipse([px-3, py-3, px+3, py+3], fill="yellow")

                if pts:
                    x, y = pts[0]
                    draw.text(
                        (x + 5, y - 15),
                        f"{cls} (pose)",
                        fill="white",
                        font=font,
                        stroke_width=3,
                        stroke_fill="black"
                    )

        # === Guardar overlay final ===
        final = img.convert("RGB")
        overlay_name = unique_name(".jpg")
        final.save(os.path.join(storage_dir, overlay_name), "JPEG", quality=90)

        # === Registrar ===
        res = EtiquetaManualResult(
            asset_id=asset.id,
            annotations=ann,
            overlay_filename=overlay_name
        )

        db.session.add(res)
        saved.append(asset_id)

    db.session.commit()

    return jsonify({
        "saved": True,
        "assets_saved": saved,
        "count": len(saved)
    }), 201
