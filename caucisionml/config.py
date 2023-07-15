from pydantic import BaseSettings
from os import environ
import socket

environment = environ.get("FASTAPI_ENV")


class Settings(BaseSettings):
    class Config:
        # Order: prod > development > test
        env_file = ".env.test", ".env.development", ".env.production"
        case_sensitive = False

    app_name: str = "CaucisionML"

    DATABASE_URL: str
    SCYLLA_HOST: str = socket.gethostbyname(socket.gethostname())
    SCYLLA_KEYSPACE: str = "caucision"
    REDIS_URL: str
    CELERY_BROKER_URL: str
    API_GATEWAY_URL: str


settings = Settings()
