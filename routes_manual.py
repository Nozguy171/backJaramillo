from flask import Blueprint, request, jsonify, current_app, send_from_directory
from flask_jwt_extended import jwt_required, get_jwt_identity
from extensions import db
from models import (
    EtiquetaManualAsset,
    EtiquetaManualResult,
    Project,
    Usuario
)
from utils.github_upload import github_upload
from utils.project_classes import get_or_create_class_id
from utils.github_project_structure import create_project_structure
from utils_storage import save_pil_image, save_upload_file, unique_name
from PIL import Image
import os, io, base64
import os
from PIL import Image, ImageDraw
from PIL import ImageFont
from flask import send_file  
from slugify import slugify

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
    project_id = request.form.get("project_id")

    if project_id:
        # Validar proyecto pertenece al usuario
        proj = Project.query.filter_by(id=project_id, user_id=user_id).first()
        if not proj:
            return jsonify({"error": "Proyecto inválido"}), 400

    img = Image.open(up.stream).convert("RGB")
    width, height = img.size

    filename, path = save_pil_image(img, storage_dir, ext=".jpg", quality=92)

    row = EtiquetaManualAsset(
        user_id=user_id,
        project_id=project_id,
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
        "height": height,
        "project_id": project_id
    }), 201

# =====================
# 2. LISTAR IMÁGENES
# =====================
@bp_manual.get("/images")
@jwt_required()
def manual_images_list():
    user_id = int(get_jwt_identity())
    project_id = request.args.get("project_id")

    q = EtiquetaManualAsset.query.filter_by(user_id=user_id)

    if project_id:
        q = q.filter_by(project_id=project_id)

    rows = q.order_by(EtiquetaManualAsset.created_at.desc()).all()

    out = []
    for r in rows:
        out.append({
            "id": r.id,
            "url": public_url(r.filename),
            "width": r.width,
            "height": r.height,
            "created_at": r.created_at.isoformat(),
            "project_id": r.project_id
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
        "project_id": asset.project_id,   # 🔥 AQUI
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

    user = Usuario.query.get(user_id)
    proj = Project.query.get(asset.project_id)

    if proj:
        user_folder = user.correo.replace("@", "_at_")
        proj_folder = slugify(proj.name)

        # rutas a crear en el repo
        repo_img_path = f"{user_folder}/{proj_folder}/images/{asset.id}.jpg"
        repo_label_path = f"{user_folder}/{proj_folder}/labels/{asset.id}.txt"

        # subir imagen original (o overlay si quieres)
        github_upload(repo_img_path, img_path)

        # generar TXT estilo YOLO
        W, H = asset.width, asset.height
        label_lines = []

        for a in ann:
            t = a.get("type")
            cls = a.get("class", "obj")
            cid = get_or_create_class_id(asset.project_id, cls)
            # ==== BBOX ====
            if t == "bbox":
                x1, y1, x2, y2 = a["bbox"]
                cx = (x1 + x2) / 2 / W
                cy = (y1 + y2) / 2 / H
                bw = abs(x2 - x1) / W
                bh = abs(y2 - y1) / H
                label_lines.append(f"{cid} {cx} {cy} {bw} {bh}")
                continue

            # ==== OBB ====
            if t == "obb":
                pts = a.get("points", [])
                if not pts or len(pts) < 4:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = sum(xs) / len(xs) / W
                cy = sum(ys) / len(ys) / H
                bw = (max(xs) - min(xs)) / W
                bh = (max(ys) - min(ys)) / H
                ang = a.get("angle", 0)
                label_lines.append(f"{cid} {cx} {cy} {bw} {bh} {ang}")
                continue

            # ==== SEG ====
            if t == "seg":
                pts = a.get("points", [])
                if not pts or len(pts) < 3:
                    continue

                flat = []
                for px, py in pts:
                    flat.append(f"{px/W} {py/H}")

                label_lines.append(f"{cid} " + " ".join(flat))
                continue

            # ==== POSE ====
            if t == "pose":
                pts = a.get("points", [])
                if not pts:
                    continue

                flat = []
                for px, py in pts:
                    flat.extend([f"{px/W}", f"{py/H}", "2"])  # visibilidad = 2

                label_lines.append(f"{cid} " + " ".join(flat))
                continue

        label_txt = "\n".join(label_lines)


        tmp_path = f"/tmp/{asset.id}.txt"
        with open(tmp_path, "w") as f:
            f.write(label_txt)

        github_upload(repo_label_path, tmp_path)
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
            "project_id": r.project_id,
        })

    return jsonify(out), 200
@bp_manual.post("/batch/save")
@jwt_required()
def manual_batch_save():
    """
    Guarda batch completo de anotaciones,
    genera overlays, sube imagen y labels a GitHub
    igual que /save pero para muchas imágenes.
    """
    from PIL import Image, ImageDraw, ImageFont
    import os

    data = request.get_json() or {}
    items = data.get("items", [])

    if not isinstance(items, list):
        return jsonify({"error": "items debe ser lista"}), 400

    user_id = int(get_jwt_identity())
    storage_dir = current_app.config["STORAGE_DIR"]

    # ============================
    # 🔥 VALIDACIÓN PREVIA GLOBAL
    # ============================
    for item in items:
        asset_id = item.get("id")
        ann = item.get("annotations")

        if not asset_id:
            return jsonify({"error": "Falta id en un item del batch"}), 400

        if not isinstance(ann, list):
            return jsonify({"error": f"La imagen {asset_id} no tiene anotaciones válidas."}), 400

        if len(ann) == 0:
            return jsonify({"error": f"La imagen {asset_id} no tiene anotaciones. Debes completar todas antes de guardar."}), 400

    saved = []

    # ============================
    # 🔥 PROCESAR TODAS LAS IMÁGENES
    # ============================
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

        # === Cargar imagen original
        img_path = os.path.join(storage_dir, asset.filename)
        img = Image.open(img_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        # ============================
        # 🔥 DIBUJAR ANOTACIONES
        # ============================
        for a in ann:
            t = a.get("type")
            cls = a.get("class", "")

            # ---- BBOX ----
            if t == "bbox":
                x1, y1, x2, y2 = a["bbox"]
                x0, y0 = min(x1, x2), min(y1, y2)
                x1, y1 = max(x1, x2), max(y1, y2)

                draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                draw.text(
                    (x0 + 5, y0 - 15),
                    f"{cls} (bbox)",
                    fill="white", font=font,
                    stroke_width=3, stroke_fill="black"
                )

            # ---- OBB ----
            elif t == "obb":
                pts = a.get("points", [])
                if pts:
                    draw.polygon(pts, outline="orange", width=3)
                    x, y = pts[0]
                    draw.text(
                        (x + 5, y - 15),
                        f"{cls} (obb)",
                        fill="white", font=font,
                        stroke_width=3, stroke_fill="black"
                    )

            # ---- SEG ----
            elif t == "seg":
                pts = a.get("points", [])
                if pts:
                    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    d2 = ImageDraw.Draw(ov)
                    d2.polygon(pts, fill=(0, 255, 255, 80), outline=(0, 180, 255, 255))
                    img = Image.alpha_composite(img, ov)
                    draw = ImageDraw.Draw(img)
                    x, y = pts[0]
                    draw.text(
                        (x + 5, y - 15),
                        f"{cls} (seg)",
                        fill="white", font=font,
                        stroke_width=3, stroke_fill="black"
                    )

            # ---- POSE ----
            elif t == "pose":
                pts = a.get("points", [])
                for (px, py) in pts:
                    draw.ellipse([px-3, py-3, px+3, py+3], fill="yellow")

                if pts:
                    x, y = pts[0]
                    draw.text(
                        (x + 5, y - 15),
                        f"{cls} (pose)",
                        fill="white", font=font,
                        stroke_width=3, stroke_fill="black"
                    )

        # ============================
        # 🔥 GUARDAR OVERLAY
        # ============================
        final = img.convert("RGB")
        overlay_name = unique_name(".jpg")
        final.save(os.path.join(storage_dir, overlay_name), "JPEG", quality=90)

        res = EtiquetaManualResult(
            asset_id=asset.id,
            annotations=ann,
            overlay_filename=overlay_name
        )
        db.session.add(res)

        # ============================
        # 🔥 SUBIR A GITHUB (IMAGEN + LABELS)
        # ============================
        user = Usuario.query.get(user_id)
        proj = Project.query.get(asset.project_id)

        if proj:
            from slugify import slugify

            user_folder = user.correo.replace("@", "_at_")
            proj_folder = slugify(proj.name)

            repo_img_path = f"{user_folder}/{proj_folder}/images/{asset.id}.jpg"
            repo_label_path = f"{user_folder}/{proj_folder}/labels/{asset.id}.txt"

            # 1) Subir imagen
            github_upload(repo_img_path, img_path)

            # 2) Generar TXT YOLO
            W, H = asset.width, asset.height
            label_lines = []

            for a in ann:
                t = a.get("type")
                cls = a.get("class", "obj")
                cid = get_or_create_class_id(asset.project_id, cls)

                # --- BBOX ---
                if t == "bbox":
                    x1, y1, x2, y2 = a["bbox"]
                    cx = (x1 + x2) / 2 / W
                    cy = (y1 + y2) / 2 / H
                    bw = abs(x2 - x1) / W
                    bh = abs(y2 - y1) / H
                    label_lines.append(f"{cid} {cx} {cy} {bw} {bh}")
                    continue

                # --- OBB ---
                if t == "obb":
                    pts = a.get("points", [])
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    cx = sum(xs)/len(xs)/W
                    cy = sum(ys)/len(ys)/H
                    bw = (max(xs)-min(xs))/W
                    bh = (max(ys)-min(ys))/H
                    ang = a.get("angle", 0)
                    label_lines.append(f"{cid} {cx} {cy} {bw} {bh} {ang}")
                    continue

                # --- SEG ---
                if t == "seg":
                    pts = a.get("points", [])
                    flat = [f"{px/W} {py/H}" for px, py in pts]
                    label_lines.append(f"{cid} " + " ".join(flat))
                    continue

                # --- POSE ---
                if t == "pose":
                    pts = a.get("points", [])
                    flat = []
                    for px, py in pts:
                        flat.extend([f"{px/W}", f"{py/H}", "2"])
                    label_lines.append(f"{cid} " + " ".join(flat))
                    continue

            # Guardar TMP
            tmp_path = f"/tmp/{asset.id}.txt"
            with open(tmp_path, "w") as f:
                f.write("\n".join(label_lines))

            # Subir TXT
            github_upload(repo_label_path, tmp_path)

        saved.append(asset_id)

    # TERMINAR
    db.session.commit()

    return jsonify({
        "saved": True,
        "assets_saved": saved,
        "count": len(saved)
    }), 201

@bp_manual.post("/export/yolo/batch")
@jwt_required()
def export_yolo_batch():
    """
    Exporta SOLO las imágenes indicadas por el usuario (batch),
    generando un ZIP compatible con YOLO.
    Maneja bbox, obb, seg y pose SIN TRONAR.
    """
    import zipfile
    import io

    data = request.get_json() or {}
    ids = data.get("ids", [])

    if not isinstance(ids, list) or len(ids) == 0:
        return jsonify({"error": "Debes enviar 'ids' como lista"}), 400

    user_id = int(get_jwt_identity())
    storage_dir = current_app.config["STORAGE_DIR"]

    mem = io.BytesIO()
    zipf = zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED)

    # Mapeo clase → ID YOLO
    classes = {}
    class_counter = 0

    def cls_id(name: str):
        nonlocal class_counter
        if name not in classes:
            classes[name] = class_counter
            class_counter += 1
        return classes[name]

    exported = 0

    for asset_id in ids:

        asset = (
            EtiquetaManualAsset.query
            .filter_by(id=asset_id, user_id=user_id)
            .first()
        )

        if not asset:
            continue

        last = (
            EtiquetaManualResult.query
            .filter_by(asset_id=asset.id)
            .order_by(EtiquetaManualResult.created_at.desc())
            .first()
        )

        if not last or not last.annotations:
            continue

        W, H = asset.width, asset.height
        img_path = os.path.join(storage_dir, asset.filename)

        if not os.path.exists(img_path):
            continue

        zipf.write(img_path, f"images/{asset_id}.jpg")

        label_lines = []

        for ann in last.annotations:
            t = ann.get("type")
            cls = ann.get("class", "obj")
            cid = cls_id(cls)

            # BBOX
            if t == "bbox":
                bbox = ann.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2 / W
                cy = (y1 + y2) / 2 / H
                bw = abs(x2 - x1) / W
                bh = abs(y2 - y1) / H

                label_lines.append(f"{cid} {cx} {cy} {bw} {bh}")
                continue

            # OBB
            if t == "obb":
                pts = ann.get("points")
                if not pts or len(pts) < 4:
                    continue

                try:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                except:
                    continue

                cx = sum(xs) / len(xs) / W
                cy = sum(ys) / len(ys) / H
                bw = (max(xs) - min(xs)) / W
                bh = (max(ys) - min(ys)) / H
                ang = ann.get("angle", 0)

                label_lines.append(f"{cid} {cx} {cy} {bw} {bh} {ang}")
                continue

            # SEG
            if t == "seg":
                pts = ann.get("points")
                if not pts or len(pts) < 3:
                    continue

                flat = []
                ok = True
                for p in pts:
                    if len(p) != 2:
                        ok = False
                        break
                    flat.append(f"{p[0]/W} {p[1]/H}")
                if not ok:
                    continue

                label_lines.append(f"{cid} " + " ".join(flat))
                continue

            # POSE
            if t == "pose":
                pts = ann.get("points")
                if not pts:
                    continue

                flat = []
                ok = True
                for p in pts:
                    if len(p) != 2:
                        ok = False
                        break
                    flat.extend([f"{p[0]/W}", f"{p[1]/H}", "2"])
                if not ok:
                    continue

                label_lines.append(f"{cid} " + " ".join(flat))
                continue

        if len(label_lines) == 0:
            continue

        zipf.writestr(f"labels/{asset_id}.txt", "\n".join(label_lines))
        exported += 1

    # dataset.yaml
    yaml = "path: .\ntrain: images\nval: images\nnames:\n"
    for cls, cid in classes.items():
        yaml += f"  {cid}: {cls}\n"

    zipf.writestr("dataset.yaml", yaml)

    zipf.close()
    mem.seek(0)

    if exported == 0:
        return jsonify({"error": "Ninguna imagen válida para exportar"}), 400

    return send_file(
        mem,
        mimetype="application/zip",
        as_attachment=True,
        download_name="yolo_batch_dataset.zip"
    )

@bp_manual.post("/projects")
@jwt_required()
def create_project():
    user_id = int(get_jwt_identity())
    data = request.get_json() or {}

    name = data.get("name")
    if not name:
        return jsonify({"error": "nombre requerido"}), 400

    slug = slugify(name)

    p = Project(user_id=user_id, name=name, slug=slug)
    db.session.add(p)
    db.session.commit()

    # === Crear estructura del proyecto en GitHub ===
    u = Usuario.query.get(user_id)
    create_project_structure(u.correo, slug)

    return jsonify({
        "id": p.id,
        "name": p.name,
        "slug": p.slug
    }), 201


@bp_manual.get("/projects")
@jwt_required()
def list_projects():
    user_id = int(get_jwt_identity())

    rows = Project.query.filter_by(user_id=user_id).order_by(Project.created_at.desc()).all()

    return jsonify([
        {
            "id": p.id,
            "name": p.name,
            "slug": p.slug,
            "created_at": p.created_at.isoformat()
        }
        for p in rows
    ]), 200

@bp_manual.post("/projects/<int:project_id>/classes")
@jwt_required()
def create_class(project_id):
    user_id = int(get_jwt_identity())
    data = request.get_json() or {}

    name = data.get("name")
    if not name:
        return jsonify({"error": "name requerido"}), 400

    # verificar que el proyecto es del usuario
    proj = Project.query.filter_by(id=project_id, user_id=user_id).first()
    if not proj:
        return jsonify({"error": "Proyecto inválido"}), 400

    cid = get_or_create_class_id(project_id, name)

    return jsonify({
        "name": name,
        "class_id": cid
    }), 201

from models import ProjectClass

@bp_manual.get("/projects/<int:project_id>/classes")
@jwt_required()
def list_classes(project_id):
    user_id = int(get_jwt_identity())

    proj = Project.query.filter_by(id=project_id, user_id=user_id).first()
    if not proj:
        return jsonify({"error": "Proyecto inválido"}), 400

    rows = ProjectClass.query.filter_by(project_id=project_id).order_by(ProjectClass.class_id.asc()).all()

    return jsonify([
        {"name": r.name, "class_id": r.class_id}
        for r in rows
    ]), 200
