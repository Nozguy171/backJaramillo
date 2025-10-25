import os
import base64
from typing import Tuple, List, Dict, Any, Optional, Callable

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def bgr_to_b64jpg(bgr: np.ndarray, quality: int = 90) -> Optional[str]:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8") if ok else None

def pngmask_to_b64(mask01: np.ndarray) -> Optional[str]:
    ok, buf = cv2.imencode(".png", (mask01.astype(np.uint8) * 255))
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8") if ok else None

def resize_mask_to_image(mask01: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    H, W = img_shape[:2]
    if mask01.dtype != np.uint8:
        mask01 = mask01.astype(np.uint8)
    mask_resized = cv2.resize(mask01, (W, H), interpolation=cv2.INTER_NEAREST)
    return (mask_resized > 0).astype(np.uint8)

def mask_to_polygon_and_obb(mask01: np.ndarray):
    m = (mask01 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], None
    cnt = max(contours, key=cv2.contourArea)
    eps = 0.01 * cv2.arcLength(cnt, True)
    poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).tolist()
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).tolist()
    (cx, cy), (w, h), angle = rect
    obb = {
        "center": [float(cx), float(cy)],
        "size": [float(w), float(h)],
        "angle": float(angle),
        "box": [[float(x), float(y)] for x, y in box],
    }
    return poly, obb

def _color_for_class(name: str) -> tuple[int, int, int]:
    import hashlib
    h = hashlib.md5(name.encode()).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    boost = lambda x: int(64 + (x % 192))
    return (boost(b), boost(g), boost(r))

def _draw_mask_with_outline(dst: np.ndarray, mask01: np.ndarray, color=(0, 255, 0),
                            alpha: float = 0.5, outline: int = 3):
    """
    Aplica color con alpha y dibuja contorno doble (negro + color) para que resalte.
    """
    H, W = dst.shape[:2]
    if mask01.ndim == 3:
        mask01 = mask01.squeeze()
    if mask01.shape != (H, W):
        mask01 = cv2.resize(mask01.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    m = (mask01 > 0).astype(np.uint8)

    color_layer = np.zeros_like(dst, dtype=np.uint8)
    color_layer[:] = color
    blended = cv2.addWeighted(color_layer, alpha, dst, 1 - alpha, 0)
    out = dst.copy()
    out[m == 1] = blended[m == 1]

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(out, cnts, -1, (0, 0, 0), thickness=outline + 2, lineType=cv2.LINE_AA)
        cv2.drawContours(out, cnts, -1, color, thickness=outline, lineType=cv2.LINE_AA)
    return out

def apply_overlay(bgr: np.ndarray, mask01: np.ndarray, color=(0, 255, 0), alpha=0.5) -> np.ndarray:
    return _draw_mask_with_outline(bgr, mask01, color=color, alpha=alpha, outline=3)

def _aabb_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def _quad_to_aabb(quad: List[List[float]]) -> List[float]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return [min(xs), min(ys), max(xs), max(ys)]

def _best_writer(path_out: str, w: int, h: int, fps: float):
    if w % 2 != 0: w -= 1
    if h % 2 != 0: h -= 1
    if fps is None or fps <= 0: fps = 25.0

    avi1 = os.path.splitext(path_out)[0] + ".avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(avi1, fourcc, fps, (w, h))
    if writer.isOpened():
        return writer, avi1

    avi2 = os.path.splitext(path_out)[0] + "_xvid.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(avi2, fourcc, fps, (w, h))
    if writer.isOpened():
        return writer, avi2

    mp4 = os.path.splitext(path_out)[0] + ".mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4, fourcc, fps, (w, h))
    if writer.isOpened():
        return writer, mp4

    return writer, None

def _count_objects(objs):
    c = {}
    for o in objs:
        k = o.get("class", "obj")
        c[k] = c.get(k, 0) + 1
    return c

_COCO_SKELETON = [
    (5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),(0,5),(0,6)
]

def _draw_pose(overlay: np.ndarray, kpts: np.ndarray, conf_th: float = 0.25):
    K = kpts.shape[0]
    for i in range(K):
        x, y, c = kpts[i]
        if c >= conf_th:
            cv2.circle(overlay, (int(x), int(y)), 3, (255, 255, 0), -1)
    for a, b in _COCO_SKELETON:
        if a < K and b < K:
            xa, ya, ca = kpts[a]
            xb, yb, cb = kpts[b]
            if ca >= conf_th and cb >= conf_th:
                cv2.line(overlay, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 255), 2)

def _boxes_from_keypoints(kpts: np.ndarray) -> Optional[List[float]]:
    valid = kpts[:, 2] > 0.05
    if not np.any(valid):
        return None
    xs = kpts[valid, 0]; ys = kpts[valid, 1]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

def _draw_obb_quad(img: np.ndarray, quad: List[List[float]], color=(0, 200, 255), thick=3):
    pts = np.array(quad, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(0,0,0), thickness=thick+2, lineType=cv2.LINE_AA)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thick,   lineType=cv2.LINE_AA)

def _obb_from_ultralytics_result(r0) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        names = getattr(r0, "names", {}) or {}
        has_obb = hasattr(r0, "obb") and r0.obb is not None
        if not has_obb:
            return out

        quads = getattr(r0.obb, "xyxyxyxy", None)
        xywhr = getattr(r0.obb, "xywhr", None)

        if quads is not None:
            quads = quads.cpu().numpy() 
            cls = r0.obb.cls.cpu().numpy().astype(int) if hasattr(r0.obb, "cls") else None
            conf = r0.obb.conf.cpu().numpy() if hasattr(r0.obb, "conf") else None
            for i in range(quads.shape[0]):
                pts = quads[i].reshape(-1, 2).tolist()
                item = {
                    "class": names.get(int(cls[i]), str(int(cls[i]))) if cls is not None else "obj",
                    "score": float(conf[i]) if conf is not None else 1.0,
                    "quad": [[float(x), float(y)] for x, y in pts],
                }
                if xywhr is not None:
                    cx, cy, w, h, t = xywhr[i].tolist()
                    item["xywhr"] = [float(cx), float(cy), float(w), float(h), float(t)]
                out.append(item)

        elif xywhr is not None:
            xywhr = xywhr.cpu().numpy()  
            cls = r0.obb.cls.cpu().numpy().astype(int) if hasattr(r0.obb, "cls") else None
            conf = r0.obb.conf.cpu().numpy() if hasattr(r0.obb, "conf") else None
            for i in range(xywhr.shape[0]):
                cx, cy, w, h, t = xywhr[i].tolist()
                angle = float(t)
                if abs(angle) <= 3.14159: 
                    angle_deg = angle * 180.0 / 3.14159
                else:
                    angle_deg = angle
                rect = ((cx, cy), (w, h), angle_deg)
                quad = cv2.boxPoints(rect).tolist()
                out.append({
                    "class": names.get(int(cls[i]), str(int(cls[i]))) if cls is not None else "obj",
                    "score": float(conf[i]) if conf is not None else 1.0,
                    "quad": [[float(x), float(y)] for x, y in quad],
                    "xywhr": [float(cx), float(cy), float(w), float(h), float(angle_deg)],
                })
    except Exception as e:
        print("[obb] ⚠️ no se pudieron parsear OBBs:", e)
    return out

class YoloSegmenter:
    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
        seg_weights = (
            weights_path
            or os.getenv("SEG_MODEL_PATH")
            or os.getenv("VISION_MODEL_PATH")
            or "/app/models/best_seg.pt"
        )
        pose_weights = os.getenv("POSE_MODEL_PATH") or "/app/models/best_pose.pt"
        obb_weights  = os.getenv("OBB_MODEL_PATH")  or "/app/models/best_obb.pt"

        print(f"[yolo] 🔧 cargando SEG desde:  {seg_weights}")
        self.model = YOLO(seg_weights)
        self.device = device

        self.pose_model: Optional[YOLO] = None
        try:
            if pose_weights and os.path.exists(pose_weights):
                print(f"[yolo] 🔧 cargando POSE desde: {pose_weights}")
                self.pose_model = YOLO(pose_weights)
        except Exception as e:
            print(f"[yolo] ⚠️ no se pudo cargar POSE: {e}")

        self.obb_model: Optional[YOLO] = None
        try:
            if obb_weights and os.path.exists(obb_weights):
                print(f"[yolo] 🔧 cargando OBB  desde: {obb_weights}")
                self.obb_model = YOLO(obb_weights)
        except Exception as e:
            print(f"[yolo] ⚠️ no se pudo cargar OBB: {e}")

        print(f"[yolo] ✅ modelos listos (device={self.device}, pose={'on' if self.pose_model else 'off'}, obb={'on' if self.obb_model else 'off'})")

    def segment_image_bgr(self, bgr: np.ndarray, conf: float = 0.3, imgsz: int = 640) -> Dict[str, Any]:
        results = self.model.predict(source=[bgr], conf=conf, imgsz=imgsz, device=self.device, verbose=False)
        overlay = bgr.copy()
        objects: List[Dict[str, Any]] = []

        if results:
            r0 = results[0]
            names = getattr(r0, "names", getattr(self.model, "names", {})) or {}
            boxes = r0.boxes
            masks = r0.masks
            num = len(boxes) if boxes is not None else 0

            for i in range(num):
                cls_id = int(boxes.cls[i])
                cls_name = names.get(cls_id, str(cls_id))
                score = float(boxes.conf[i])
                x1, y1, x2, y2 = map(float, boxes.xyxy[i].tolist())

                polygon, obb = None, None
                mask_b64 = None
                color_cls = _color_for_class(cls_name)

                if masks is not None and i < len(masks.data):
                    mask = masks.data[i].cpu().numpy()
                    mask01 = (mask > 0.5).astype(np.uint8)
                    mask01 = resize_mask_to_image(mask01, bgr.shape)
                    polygon, obb = mask_to_polygon_and_obb(mask01)
                    mask_b64 = pngmask_to_b64(mask01)
                    overlay = _draw_mask_with_outline(overlay, mask01, color=color_cls, alpha=0.5, outline=3)

                x1i, y1i = int(x1), int(y1)
                label = f"{cls_name} {int(score*100)}%"
                cv2.putText(overlay, label, (x1i, max(12, y1i - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(overlay, label, (x1i, max(12, y1i - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                objects.append({
                    "id": i,
                    "class": cls_name,
                    "score": score,
                    "bbox": [x1, y1, x2, y2],
                    "polygon": polygon,
                    "obb": obb,
                    "mask_b64": mask_b64,
                })

        if self.obb_model is not None:
            obb_res = self.obb_model.predict(source=[bgr], conf=conf, imgsz=imgsz, device=self.device, verbose=False)
            if obb_res:
                d = _obb_from_ultralytics_result(obb_res[0])
                for det in d:
                    quad = det["quad"]
                    _draw_obb_quad(overlay, quad) 
                    aabb = _quad_to_aabb(quad)
                    best_j, best_iou = -1, 0.0
                    for j, o in enumerate(objects):
                        iou = _aabb_iou(o["bbox"], aabb)
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    if best_j >= 0 and best_iou > 0.1:
                        objects[best_j]["obb_det"] = {
                            "quad": [[float(x), float(y)] for x, y in quad],
                            "aabb": aabb,
                            "score": det.get("score", 1.0),
                            "class": det.get("class", "obj"),
                            "xywhr": det.get("xywhr"),
                            "iou_with_det": best_iou
                        }

        if self.pose_model is not None:
            pres = self.pose_model.predict(source=[bgr], conf=conf, imgsz=imgsz, device=self.device, verbose=False)
            if pres:
                p0 = pres[0]
                if getattr(p0, "keypoints", None) is not None and p0.keypoints is not None:
                    karr = p0.keypoints.data.cpu().numpy()
                    for kp in karr:
                        bbox_p = _boxes_from_keypoints(kp)
                        _draw_pose(overlay, kp)
                        if bbox_p is None or not objects:
                            continue
                        best_j, best_iou = -1, 0.0
                        for j, o in enumerate(objects):
                            iou = _aabb_iou(o["bbox"], bbox_p)
                            if iou > best_iou:
                                best_iou, best_j = iou, j
                        if best_j >= 0 and best_iou > 0.1:
                            objects[best_j]["pose"] = {
                                "kpts": [[float(x), float(y), float(c)] for (x, y, c) in kp.tolist()],
                                "bbox_from_kpts": bbox_p,
                                "iou_with_det": best_iou,
                            }

        return {"objects": objects, "overlay_jpg_b64": bgr_to_b64jpg(overlay)}

    def segment_pil(self, img: Image.Image, **kw) -> Dict[str, Any]:
        bgr = pil_to_bgr(img)
        return self.segment_image_bgr(bgr, **kw)

    def segment_video_path(
        self,
        in_path: str,
        out_path: str,
        conf: float = 0.25,
        imgsz: int = 640,
        sample_every: int = 1,
        max_side: int | None = 1280,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ):
        print(f"[segment/video] ▶️ abriendo {in_path}")
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir el video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = float(N) / float(fps) if fps > 0 else 0.0
        print(f"[segment/video] meta: {W}x{H} @ {fps:.2f}fps, frames={N}, dur={duration:.2f}s")

        scale = 1.0
        if max_side and max(W, H) > max_side:
            scale = max_side / float(max(W, H))
        Ww = int(round(W * scale))
        Hh = int(round(H * scale))
        if Ww % 2 != 0: Ww -= 1
        if Hh % 2 != 0: Hh -= 1
        if fps is None or fps <= 0: fps = 25.0

        writer, final_out = _best_writer(out_path, Ww, Hh, fps)
        print(f"[segment/video] writer -> {final_out} (abierto={writer.isOpened()}) size=({Ww}x{Hh}) fps={fps}")
        if not writer.isOpened() or final_out is None:
            cap.release()
            raise RuntimeError("No se pudo abrir ningún VideoWriter (MJPG/XVID/mp4v)")

        totals: Dict[str, int] = {}
        poses_total = 0
        processed = 0
        idx = 0

        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break

                bgr_in = cv2.resize(bgr, (Ww, Hh), interpolation=cv2.INTER_AREA) if scale != 1.0 else bgr

                if sample_every > 1 and (idx % sample_every) != 0:
                    writer.write(bgr_in)
                    idx += 1
                    if progress_cb:
                        try: progress_cb(processed, N)
                        except: pass
                    continue

                results = self.model.predict(source=[bgr_in], conf=conf, imgsz=imgsz, device=self.device, verbose=False)
                overlay = bgr_in.copy()
                objs: List[Dict[str, Any]] = []
                if results:
                    r0 = results[0]
                    names = getattr(r0, "names", getattr(self.model, "names", {})) or {}
                    boxes = r0.boxes
                    masks = r0.masks
                    num = len(boxes) if boxes is not None else 0
                    for i in range(num):
                        cls_id = int(boxes.cls[i])
                        cls_name = names.get(cls_id, str(cls_id))
                        score = float(boxes.conf[i])
                        x1, y1, x2, y2 = map(float, boxes.xyxy[i].tolist())
                        polygon, obb = None, None
                        color_cls = _color_for_class(cls_name)
                        if masks is not None and i < len(masks.data):
                            mask = masks.data[i].cpu().numpy()
                            mask01 = (mask > 0.5).astype(np.uint8)
                            if mask01.shape[:2] != overlay.shape[:2]:
                                mask01 = cv2.resize(mask01, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
                            polygon, obb = mask_to_polygon_and_obb(mask01)
                            overlay = _draw_mask_with_outline(overlay, mask01, color=color_cls, alpha=0.5, outline=3)
                        x1i, y1i = int(x1), int(y1)
                        label = f"{cls_name} {int(score*100)}%"
                        cv2.putText(overlay, label, (x1i, max(12, y1i - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(overlay, label, (x1i, max(12, y1i - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                        objs.append({
                            "id": i,
                            "class": cls_name,
                            "score": score,
                            "bbox": [x1, y1, x2, y2],
                            "polygon": polygon,
                            "obb": obb,
                        })

                if self.obb_model is not None:
                    obb_res = self.obb_model.predict(source=[bgr_in], conf=conf, imgsz=imgsz, device=self.device, verbose=False)
                    if obb_res:
                        d = _obb_from_ultralytics_result(obb_res[0])
                        for det in d:
                            quad = det["quad"]
                            _draw_obb_quad(overlay, quad)
                            aabb = _quad_to_aabb(quad)
                            best_j, best_iou = -1, 0.0
                            for j, o in enumerate(objs):
                                iou = _aabb_iou(o["bbox"], aabb)
                                if iou > best_iou:
                                    best_iou, best_j = iou, j
                            if best_j >= 0 and best_iou > 0.1:
                                objs[best_j]["obb_det"] = {
                                    "quad": [[float(x), float(y)] for x, y in quad],
                                    "aabb": aabb,
                                    "score": det.get("score", 1.0),
                                    "class": det.get("class", "obj"),
                                    "xywhr": det.get("xywhr"),
                                    "iou_with_det": best_iou
                                }

                if self.pose_model is not None:
                    pres = self.pose_model.predict(source=[bgr_in], conf=conf, imgsz=imgsz, device=self.device, verbose=False)
                    if pres:
                        p0 = pres[0]
                        if getattr(p0, "keypoints", None) is not None and p0.keypoints is not None:
                            karr = p0.keypoints.data.cpu().numpy()
                            poses_total += int(karr.shape[0])
                            for kp in karr:
                                bbox_p = _boxes_from_keypoints(kp)
                                _draw_pose(overlay, kp)
                                if bbox_p is None or not objs:
                                    continue
                                best_j, best_iou = -1, 0.0
                                for j, o in enumerate(objs):
                                    iou = _aabb_iou(o["bbox"], bbox_p)
                                    if iou > best_iou:
                                        best_iou, best_j = iou, j
                                if best_j >= 0 and best_iou > 0.1:
                                    objs[best_j]["pose"] = {
                                        "kpts": [[float(x), float(y), float(c)] for (x, y, c) in kp.tolist()],
                                        "bbox_from_kpts": bbox_p,
                                        "iou_with_det": best_iou,
                                    }

                c = _count_objects(objs)
                for k, v in c.items():
                    totals[k] = totals.get(k, 0) + v

                writer.write(overlay)
                processed += 1
                idx += 1

                if processed % 50 == 0:
                    print(f"[segment/video] progreso: {processed}/{N} frames")
                if progress_cb:
                    try: progress_cb(processed, N)
                    except: pass

        finally:
            cap.release(); writer.release()

        if processed == 0:
            try:
                if final_out and os.path.exists(final_out):
                    os.remove(final_out)
            except:
                pass
            raise RuntimeError("No se escribió ningún frame en el overlay (codec/tamaño/fps)")

        print(f"[segment/video] ✅ frames procesados={processed} de {N} -> {final_out}")
        return {
            "out_path": final_out,
            "fps": fps,
            "width": Ww,
            "height": Hh,
            "frames_total": N,
            "frames_processed": processed,
            "duration_sec": duration,
            "objects_totals": totals,
            "poses_total": poses_total,
        }


seg = YoloSegmenter()
