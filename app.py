import os
from flask import Flask, jsonify, send_from_directory 
from dotenv import load_dotenv
from flask_cors import CORS
from extensions import db, migrate, jwt
from config import Config
from routes_usuario import bp_usuario
from routes_vision import bp_vision

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    if not app.config.get("SQLALCHEMY_DATABASE_URI"):
        raise RuntimeError("Falta DATABASE_URL (o está mal formada) en tu .env")

    app.config.setdefault("STORAGE_DIR", os.getenv("STORAGE_DIR", "/app/storage"))
    os.makedirs(app.config["STORAGE_DIR"], exist_ok=True)

    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)

    import models 

    CORS(
        app,
        resources={r"/api/*": {"origins": "*"}},
        supports_credentials=False,
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        expose_headers=["Content-Type"],
    )

    app.register_blueprint(bp_usuario)
    app.register_blueprint(bp_vision)

    @app.get("/storage/<path:filename>")
    def storage(filename):
        return send_from_directory(app.config["STORAGE_DIR"], filename, as_attachment=False)


    @app.get("/health")
    def health():
        return jsonify({"ok": True})

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), threaded=True)
