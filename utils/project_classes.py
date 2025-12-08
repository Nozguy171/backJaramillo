from models import ProjectClass, db

def get_or_create_class_id(project_id: int, class_name: str):
    # Buscar si existe
    row = ProjectClass.query.filter_by(project_id=project_id, name=class_name).first()
    if row:
        return row.class_id

    # Si no existe, asignar ID nuevo
    max_id = db.session.query(db.func.max(ProjectClass.class_id)).filter_by(project_id=project_id).scalar()
    next_id = 0 if max_id is None else max_id + 1

    row = ProjectClass(
        project_id=project_id,
        name=class_name,
        class_id=next_id
    )

    db.session.add(row)
    db.session.commit()
    return next_id
