import os
from datetime import timedelta

SECRET_KEY = os.getenv("SECRET_KEY", "a35f09beed6e342a7c14e3e9c9b8cfde4af7dfb7b3a823a9237c9b84783b9a4e")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour
