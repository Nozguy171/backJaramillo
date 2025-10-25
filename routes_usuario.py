from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from extensions import db
from models import Usuario

bp_usuario = Blueprint("usuario", __name__, url_prefix="/api/usuarios")

@bp_usuario.post("")
def crear_usuario():
    data = request.get_json(force=True, silent=True) or {}
    correo = data.get("correo")
    password = data.get("password")

    if not correo or not password:
        return jsonify({"error": "correo y password son requeridos"}), 400

    if Usuario.query.filter_by(correo=correo).first():
        return jsonify({"error": "correo ya registrado"}), 409

    user = Usuario(correo=correo)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return jsonify(user.to_dict()), 201

@bp_usuario.post("/login")
def login():
    data = request.get_json(force=True, silent=True) or {}
    correo = data.get("correo")
    password = data.get("password")
    if not correo or not password:
        return jsonify({"error": "correo y password son requeridos"}), 400

    u = Usuario.query.filter_by(correo=correo).first()
    if not u or not u.check_password(password):
        return jsonify({"error": "credenciales inválidas"}), 401

    token = create_access_token(identity=str(u.id))
    return jsonify({"access_token": token, "usuario": u.to_dict()}), 200

@bp_usuario.get("/me")
@jwt_required()
def me():
    user_id = int(get_jwt_identity()) 
    u = Usuario.query.get_or_404(user_id)
    return jsonify(u.to_dict()), 200


@bp_usuario.get("")
@jwt_required()
def listar_usuarios():
    users = Usuario.query.order_by(Usuario.id.desc()).all()
    return jsonify([u.to_dict() for u in users]), 200

@bp_usuario.get("/<int:user_id>")
@jwt_required()
def obtener_usuario(user_id: int):
    u = Usuario.query.get_or_404(user_id)
    return jsonify(u.to_dict()), 200

@bp_usuario.delete("/<int:user_id>")
@jwt_required()
def borrar_usuario(user_id: int):
    u = Usuario.query.get_or_404(user_id)
    db.session.delete(u)
    db.session.commit()
    return jsonify({"ok": True}), 200
