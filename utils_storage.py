import os, uuid
from PIL import Image

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def unique_name(ext: str) -> str:
    return f"{uuid.uuid4().hex}{ext}"

def save_pil_image(img: Image.Image, storage_dir: str, ext=".jpg", quality=92):
    ensure_dir(storage_dir)
    name = unique_name(ext)
    path = os.path.join(storage_dir, name)
    img.save(path, quality=quality)
    return name, path

def save_upload_file(upload_file, storage_dir: str, fallback_ext: str):
    ensure_dir(storage_dir)
    filename = upload_file.filename or ""
    _, ext = os.path.splitext(filename)
    ext = ext.lower() or fallback_ext
    name = unique_name(ext)
    path = os.path.join(storage_dir, name)
    upload_file.save(path)
    return name, path


def _best_writer(path_out: str, w: int, h: int, fps: float):
    import cv2

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
