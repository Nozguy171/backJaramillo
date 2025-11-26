from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from extensions import db
from models import (
    ImageAsset,
    VideoAsset,
    EtiquetaManualAsset
)

from datetime import datetime
import os

bp_history = Blueprint("history", __name__, url_prefix="/api/history")

def public_url(filename: str):
    base = os.getenv("PUBLIC_BACKEND_URL", "http://localhost:5051")
    return f"{base}/storage/{filename}"


@bp_history.get("")
@jwt_required()
def unified_history():
    user_id = int(get_jwt_identity())

    # --------------------------
    # AUTO IMAGE (vision)
    # --------------------------
    imgs = (
        ImageAsset.query
        .filter_by(user_id=user_id)
        .order_by(ImageAsset.created_at.desc())
        .all()
    )

    img_rows = []
    for i in imgs:
        img_rows.append({
            "kind": "image",
            "id": i.id,
            "original_url": public_url(i.filename),
            "overlay_url": public_url(i.last_result.overlay_filename) if i.last_result else None,
            "width": i.width,
            "height": i.height,
            "created_at": i.created_at.isoformat(),
            "last_result_id": i.last_result_id,
        })

    # --------------------------
    # AUTO VIDEO (vision)
    # --------------------------
    vids = (
        VideoAsset.query
        .filter_by(user_id=user_id)
        .order_by(VideoAsset.created_at.desc())
        .all()
    )

    vid_rows = []
    for v in vids:
        vid_rows.append({
            "kind": "video",
            "id": v.id,
            "original_url": public_url(v.filename),
            "overlay_url": public_url(v.last_result.overlay_filename) if v.last_result else None,
            "width": v.width,
            "height": v.height,
            "fps": v.fps,
            "duration_sec": v.duration_sec,
            "created_at": v.created_at.isoformat(),
            "last_result_id": v.last_result_id,
        })

    # --------------------------
    # MANUAL IMAGE (etiquetado)
    # --------------------------
    manual_imgs = (
        EtiquetaManualAsset.query
        .filter_by(user_id=user_id)
        .order_by(EtiquetaManualAsset.created_at.desc())
        .all()
    )

    man_rows = []
    for m in manual_imgs:
        man_rows.append({
            "kind": "manual",
            "id": m.id,
            "original_url": public_url(m.filename),
            "overlay_url": None,  # manual no genera overlay
            "width": m.width,
            "height": m.height,
            "created_at": m.created_at.isoformat(),
            "last_result_id": None,
        })

    # --------------------------
    # MERGE + SORT BY DATE
    # --------------------------
    all_rows = img_rows + vid_rows + man_rows
    all_rows.sort(key=lambda r: r["created_at"], reverse=True)

    return jsonify(all_rows), 200
