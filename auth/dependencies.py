from fastapi import APIRouter, HTTPException, Depends
from auth.password_auth import PasswordAuth
from auth.jwt_handler import create_access_token
from src.dependencies import get_user_db  # ✅ Dependency injection for DB

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register")
def register_user(
    EmpID: str,
    password: str,
    role: str = "user",              # optional: can add role selection later
    user_db=Depends(get_user_db)
):
    """
    Register a new user with hashed password.
    """
    auth = PasswordAuth()

    # Hash password using bcrypt
    hashed_pw = auth.hash_password(password).decode('utf-8')

    # Check if username already exists
    existing_user = user_db.fetch_user_by_id(EmpID)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Create new user record
    new_user = {
        "EmpID": EmpID,
        "hashed_password": hashed_pw,
        "role": role,
    }

    result = user_db.create_user(new_user)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to register user")

    return {"status": "success", "message": "User registered successfully"}


@router.post("/login")
def login_user(
    EmpID: str,
    password: str,
    user_db=Depends(get_user_db)
):
    """
    Authenticate user credentials and return JWT token.
    """
    auth = PasswordAuth()

    # Fetch user by username
    user = user_db.fetch_user_by_id(EmpID)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Validate password
    if not auth.password_authenticate(password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Generate JWT token
    token_payload = {
        "sub": EmpID,
        "role": user.get("role", "user")
    }
    token = create_access_token(token_payload)

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"EmpID": EmpID, "role": user.get("role")}
    }

