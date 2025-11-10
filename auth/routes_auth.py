from fastapi import APIRouter, HTTPException
from auth.password_auth import PasswordAuth
from auth.jwt_handler import create_access_token
from database.user_crud import get_user_by_username, create_user

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register")
def register_user(username: str, password: str):
    auth = PasswordAuth()
    hashed_pw = auth.hash_password(password)
    user = create_user(username, hashed_pw)
    return {"message": "User registered successfully"}

@router.post("/login")
def login_user(username: str, password: str):
    auth = PasswordAuth()
    user = get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not auth.password_authenticate(password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": username, "role": user["role"]})
    return {"access_token": token, "token_type": "bearer"}
