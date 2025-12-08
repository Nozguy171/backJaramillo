import os
from utils.github_upload import github_upload

def ensure_user_folder(email: str):
    """
    Crea la carpeta del usuario en GitHub si no existe.
    La representación es user_at_domain en un folder con .gitkeep.
    """

    user_folder = email.replace("@", "_at_")

    # path en el repo para el archivo vacío
    repo_path = f"{user_folder}/.gitkeep"

    try:
        # sube archivo vacío
        tmp_path = "/tmp/gitkeep"
        with open(tmp_path, "w") as f:
            f.write("")  # vacío

        github_upload(repo_path, tmp_path, message=f"create folder for {user_folder}")
        return True

    except Exception as e:
        print(f"[ensure_user_folder] error: {e}")
        return False
