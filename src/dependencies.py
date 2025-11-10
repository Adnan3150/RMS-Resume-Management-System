from fastapi import Depends
from src.DataBase import (
    database_operations_users,
    databse_operations_jd_oprimized,
    database_opeprations_resume_optimized
)
from src import load_llm
from fastapi import Depends, HTTPException, Header, Cookie, status
from auth.jwt_handler import decode_access_token

def get_current_user(
    authorization: str = Header(default=None),
    access_token: str = Cookie(default=None)
):
    """
    Extract current user info from JWT token (header or cookie)
    """
    token = None

    # 1️⃣ Check both Authorization header and cookie
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
    elif access_token and access_token.startswith("Bearer "):
        token = access_token.split(" ")[1]
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing JWT token"
        )

    # 2️⃣ Decode token safely
    try:
        user_payload = decode_access_token(token)
        return user_payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )

# --- Database connections ---
def get_jd_db():
    jd_db = databse_operations_jd_oprimized.initialize_db()
    try:
        yield jd_db
    finally:
        jd_db.close()

def get_resume_db(jd_db=Depends(get_jd_db)):
    resume_db = database_opeprations_resume_optimized.initialize_db(jd_db)
    try:
        yield resume_db
    finally:
        resume_db.close()

def get_user_db(jd_db=Depends(get_jd_db)):
    user_db = database_operations_users.initialize_db(jd_db)
    try:
        yield user_db
    finally:
        user_db.close()

# --- LLM Instance ---
def get_llm():
    llm = load_llm.load()
    yield llm
