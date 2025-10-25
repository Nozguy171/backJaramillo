import os
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()

def normalize_db_url(url: str | None):
    if not url:
        return None
    url = url.strip().strip('"').strip("'")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url

class Config:
    SQLALCHEMY_DATABASE_URI = normalize_db_url(os.getenv("DATABASE_URL"))
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.getenv("SECRET_KEY", "dev")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super-jwt-secret") 
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=8)
