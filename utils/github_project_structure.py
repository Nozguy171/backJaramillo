import os
from utils.github_upload import github_upload

def create_project_structure(email: str, project_name: str):
    """
    Crea:
        user_at_domain/project_slug/images/.gitkeep
        user_at_domain/project_slug/labels/.gitkeep
    """

    user_folder = email.replace("@", "_at_")
    project_folder = project_name

    base = f"{user_folder}/{project_folder}"

    try:
        # carpetas: images
        tmp = "/tmp/gitkeep"
        with open(tmp, "w") as f:
            f.write("")

        github_upload(f"{base}/images/.gitkeep", tmp, message="init images/")
        github_upload(f"{base}/labels/.gitkeep", tmp, message="init labels/")

        return True

    except Exception as e:
        print(f"[create_project_structure] ERROR: {e}")
        return False
