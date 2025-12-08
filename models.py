from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db

# ================== USUARIO ==================

class Usuario(db.Model):
    __tablename__ = "usuario"
    id = db.Column(db.Integer, primary_key=True)
    correo = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    creado_en = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def set_password(self, raw_password: str):
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)

    def to_dict(self):
        return {"id": self.id, "correo": self.correo, "creado_en": self.creado_en.isoformat()}


# ================== IMAGEN ==================

class ImageAsset(db.Model):
    __tablename__ = "image_assets"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("usuario.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename = db.Column(db.String(255), nullable=False)
    mime = db.Column(db.String(64), nullable=False)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    last_result_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "segmentation_results.id",
            name="fk_image_assets_last_result_id",
            use_alter=True,
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    last_result = db.relationship(
        "SegmentationResult",
        foreign_keys=[last_result_id],
        uselist=False,
        post_update=True,      
        passive_deletes=True,
        back_populates="parent_image_last",
    )

    results = db.relationship(
        "SegmentationResult",
        foreign_keys="SegmentationResult.image_id",
        back_populates="parent_image",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="SegmentationResult.created_at.desc()",
        lazy="dynamic",
    )


class SegmentationResult(db.Model):
    __tablename__ = "segmentation_results"
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(
        db.Integer,
        db.ForeignKey("image_assets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    overlay_filename = db.Column(db.String(255), nullable=False)
    objects_json = db.Column(db.JSON, nullable=False, default=list)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    parent_image = db.relationship(
        "ImageAsset",
        foreign_keys=[image_id],
        back_populates="results",
        passive_deletes=True,
    )

    parent_image_last = db.relationship(
        "ImageAsset",
        foreign_keys=[ImageAsset.last_result_id],
        viewonly=True,
        back_populates="last_result",
    )


# ================== VIDEO ==================

class VideoAsset(db.Model):
    __tablename__ = "video_assets"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("usuario.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename = db.Column(db.String(255), nullable=False)
    mime = db.Column(db.String(64), nullable=False)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    fps = db.Column(db.Float)
    duration_sec = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    last_result_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "video_segmentation_results.id",
            name="fk_video_assets_last_result_id",
            use_alter=True,
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    last_result = db.relationship(
        "VideoSegmentationResult",
        foreign_keys=[last_result_id],
        uselist=False,
        post_update=True,
        passive_deletes=True,
        back_populates="parent_video_last",
    )

    results = db.relationship(
        "VideoSegmentationResult",
        foreign_keys="VideoSegmentationResult.video_id",
        back_populates="parent_video",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="VideoSegmentationResult.created_at.desc()",
        lazy="dynamic",
    )


class VideoSegmentationResult(db.Model):
    __tablename__ = "video_segmentation_results"
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(
        db.Integer,
        db.ForeignKey("video_assets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    overlay_filename = db.Column(db.String(255), nullable=False)
    objects_json = db.Column(db.JSON, nullable=False, default=dict)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    parent_video = db.relationship(
        "VideoAsset",
        foreign_keys=[video_id],
        back_populates="results",
        passive_deletes=True,
    )

    parent_video_last = db.relationship(
        "VideoAsset",
        foreign_keys=[VideoAsset.last_result_id],
        viewonly=True,
        back_populates="last_result",
    )

class EtiquetaManualAsset(db.Model):
    __tablename__ = "manual_assets"
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer, 
        db.ForeignKey("usuario.id", ondelete="CASCADE"), 
        nullable=False
    )

    project_id = db.Column(
        db.Integer,
        db.ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=True,  # puede ser null al inicio
        index=True
    )

    filename = db.Column(db.String(255), nullable=False)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class EtiquetaManualResult(db.Model):
    __tablename__ = "manual_results"
    id = db.Column(db.Integer, primary_key=True)
    asset_id = db.Column(
        db.Integer, 
        db.ForeignKey("manual_assets.id", ondelete="CASCADE"), 
        nullable=False
    )
    annotations = db.Column(db.JSON, nullable=False)  # polígonos / cajas / puntos
    overlay_filename = db.Column(db.String(255)) 
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Project(db.Model):
    __tablename__ = "projects"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("usuario.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    name = db.Column(db.String(255), nullable=False)
    slug = db.Column(db.String(255), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # para listar imágenes por proyecto
    assets = db.relationship(
        "EtiquetaManualAsset",
        backref="project",
        cascade="all, delete-orphan"
    )

class ProjectClass(db.Model):
    __tablename__ = "project_classes"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(
        db.Integer,
        db.ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    name = db.Column(db.String(255), nullable=False)
    class_id = db.Column(db.Integer, nullable=False)  # ID numérico de YOLO

    __table_args__ = (
        db.UniqueConstraint("project_id", "name", name="uq_project_class_name"),
        db.UniqueConstraint("project_id", "class_id", name="uq_project_class_id"),
    )
