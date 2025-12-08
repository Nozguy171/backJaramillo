import base64
import requests
import os

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")  # ej: "BryanJuarezLimon/acker-datasets"

if not GITHUB_TOKEN or not GITHUB_REPO:
    raise RuntimeError("Faltan variables GITHUB_TOKEN o GITHUB_REPO en el .env")

API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents"


def github_upload(path_in_repo: str, local_path: str, message="upload"):
    """
    Sube un archivo a GitHub en path_in_repo (dentro del repo)
    """
    with open(local_path, "rb") as f:
        content = f.read()

    data = {
        "message": message,
        "content": base64.b64encode(content).decode("utf-8"),
    }

    url = f"{API_URL}/{path_in_repo}"

    res = requests.put(
        url,
        json=data,
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        },
    )

    if res.status_code not in (200, 201):
        raise Exception(f"GitHub upload failed: {res.status_code}, {res.text}")

    return res.json()
