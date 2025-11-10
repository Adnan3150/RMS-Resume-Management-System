import os
import json
import logging
from typing import Optional, List
from pathlib import Path
from datetime import datetime
import zipfile

from fastapi import FastAPI, Request, Form, HTTPException, UploadFile, File, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse,HTMLResponse,RedirectResponse
from fastapi.templating import Jinja2Templates

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from src.DataBase import database_operations_users, databse_operations_jd_oprimized, database_opeprations_resume_optimized
from src.AGENTS import resume_field_agent, resume_scorer_agent
from src import agent_executer, resume_agent_executor, load_llm

from auth.routes_auth import router as auth_router
from auth.dependencies import get_current_user, role_required

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Resume Processing API",
    description="Production-ready API for processing job descriptions and resumes",
    version="1.0.0"
)
app.include_router(auth_router)
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File upload configuration
BASE_UPLOAD_FOLDER = "uploads"
RESUME_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, "resumes")
JD_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, "jds")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".zip", ".txt"}

# Create upload directories
os.makedirs(RESUME_FOLDER, exist_ok=True)
os.makedirs(JD_FOLDER, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=BASE_UPLOAD_FOLDER), name="uploads")
# Mount the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates folder setup
templates = Jinja2Templates(directory="templates")

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "error_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unexpected errors"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "An internal server error occurred",
            "error_code": 500
        }
    )


# Startup and shutdown events
@app.on_event("startup")
def startup_event():
    """Initialize application state on startup"""
    try:
        logger.info("Initializing application...")
        
        # Load LLM
        app.state.llm = load_llm.load()
        logger.info("LLM loaded successfully")
        
        # Initialize database connections
        app.state.jd_db_ops = databse_operations_jd_oprimized.initialize_db()
        app.state.resume_db_ops = database_opeprations_resume_optimized.initialize_db(app.state.jd_db_ops)
        app.state.users_db_ops=database_operations_users.initialize_db(app.state.jd_db_ops)
        logger.info("Database connections initialized")
        
        # Initialize state variables
        app.state.resumes = None
        app.state.resume_id_list = []
        app.state.jd_id_for_resume = None
        
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise


@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on application shutdown"""
    try:
        logger.info("Shutting down application...")
        
        if hasattr(app.state, 'jd_db_ops'):
            app.state.jd_db_ops.close()
        
        if hasattr(app.state, 'resume_db_ops'):
            app.state.resume_db_ops.close()
        
        logger.info("Application shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Utility functions
def validate_file_size(file: UploadFile) -> bool:
    """Validate file size"""
    try:
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        return size <= MAX_FILE_SIZE
    except Exception as e:
        logger.error(f"Error checking file size: {e}")
        return False


def is_valid_upload(file: Optional[UploadFile]) -> bool:
    """Validate uploaded file"""
    if not file or not file.filename or not file.filename.strip():
        return False

    ext = os.path.splitext(file.filename.lower())[1]
    
    if ext not in ALLOWED_EXTENSIONS:
        return False
    
    if not validate_file_size(file):
        return False
    
    return True


def save_uploaded_file(upload_folder: str, file: UploadFile) -> str:
    """Save uploaded file with timestamped name"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in '._- ')
        filename = f"{timestamp}_{safe_filename}"
        file_path = os.path.join(upload_folder, filename)
        
        with open(file_path, "wb") as f:
            content = file.file.read()
            f.write(content)
        
        logger.info(f"✅ File saved: {filename}")
        return file_path
        
    except Exception as e:
        logger.error(f"Error saving file {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )


def extract_zip(file_path: str, extract_to: str) -> List[str]:
    """Extract ZIP file and return list of extracted file paths"""
    try:
        os.makedirs(extract_to, exist_ok=True)
        extracted = []
        
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        
        for root, _, files in os.walk(extract_to):
            for f_name in files:
                file_path_full = os.path.join(root, f_name)
                # Only add valid file types
                if os.path.splitext(f_name.lower())[1] in ALLOWED_EXTENSIONS:
                    extracted.append(file_path_full)
        
        # Remove ZIP after extraction
        if os.path.exists(file_path):
            os.remove(file_path)
        
        logger.info(f"Extracted {len(extracted)} files from ZIP")
        return extracted
        
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid ZIP file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or corrupted ZIP file"
        )
    except Exception as e:
        logger.error(f"Error extracting ZIP: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract ZIP file: {str(e)}"
        )


# # API Endpoints
# @app.get("/", tags=["Health"])
# def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "message": "Resume Processing API is running",
#         "version": "1.0.0"
#     }
@app.get("/")
async def home(request: Request):
    return RedirectResponse(url=f"http://192.168.5.142:8000/login", status_code=303)

@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})



@app.get("/signup", response_class=HTMLResponse)
async def get_signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
async def create_account(
    request: Request,  # <== Add this!
    username: str = Form(...),
    email: str = Form(...),
    EmpID: str = Form(...),
    role: str= Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...)
):
    if password != confirm_password:
        return templates.TemplateResponse("signup.html", {
            "request": request,  # <== use instance here
            "error": "Passwords do not match."
        })
    role_id=next((roles["role_id"] for roles in app.state.users_db_ops.fetch_all_roles() if roles["role_name"] ==role), None)
    if role_id:
        user_data = {
            "employee_id":int(EmpID),
            "username": username,
            "email": email,
            "password_hash": password,
            "role_id": role_id
        }
    
    success = app.state.users_db_ops.insert_user(**user_data)
    if success:
        return RedirectResponse(url="/login", status_code=303)
    else:
        return templates.TemplateResponse("signup.html", {
            "request": request, 
            "error": "Signup failed. Email or username might already exist."
        })

@app.post("/login")
async def login(request: Request, EmpID: str = Form(...), password: str = Form(...)):
    try:
        user_data=app.state.users_db_ops.fetch_user_by_id(int(EmpID))
        if int(EmpID) == user_data.get("employee_id") and password == user_data.get('password_hash'):
            # Correct external redirect (remove the leading slash)
            # user_id = user_data.get("user_id")
            if user_data:
                app.state.user_id=user_data.get("employee_id")
                print(user_data)
                return RedirectResponse(url=f"http://192.168.5.142:8000/index", status_code=303)
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})
    
    except Exception as e:
        print(f"Error while validation:{e}")
        return templates.TemplateResponse("error.html", {
            "request": request, 
            "error": "Something went wrong please try again after some time"
        })

@app.get("/index", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload_resume", response_class=HTMLResponse)
def upload_resume_page(request: Request):
    return templates.TemplateResponse("upload_resume.html", {"request": request})


@app.get("/upload_jd", response_class=HTMLResponse)
def upload_jd_page(request: Request):
    return templates.TemplateResponse("upload_jd.html", {"request": request})

@app.post("/jd/dump", status_code=status.HTTP_201_CREATED, tags=["Job Description"])
async def insert_jd(request: Request):
    """Insert new Job Description"""
    try:
        data = await request.json()
        
        if not data or "jd_fields" not in data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request: missing 'jd_fields'"
            )
        jd_id = app.state.jd_db_ops.create(data["jd_fields"])
        
        if not jd_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to insert job description"
            )
        
        logger.info(f"JD inserted with ID: {jd_id}")
        return {
            "status": "success",
            "message": "Job description uploaded successfully",
            "jd_id": jd_id
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error inserting JD: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to insert job description"
        )


@app.get("/jd/fetch", tags=["Job Description"])
def fetch_jd(jd_id: int):
    """Fetch Job Description by ID"""
    try:
        if jd_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid jd_id: must be positive integer"
            )
        
        result = app.state.jd_db_ops.read(jd_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job description not found for jd_id: {jd_id}"
            )
        
        logger.info(f"JD fetched: {jd_id}")
        return {
            "status": "success",
            "message": "Job description fetched successfully",
            "details": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching JD {jd_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch job description"
        )


@app.get("/jd/extract_fields", tags=["Job Description"])
def extract_jd_fields(jd_text: str):
    """Extract fields from JD text using AI"""
    try:
        if not jd_text or not jd_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="JD text cannot be empty"
            )
        
        graph = agent_executer.build_graph(app.state.llm)
        result = agent_executer.execute(graph, jd_text)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to extract JD fields"
            )
        jd_json=result
        jd_json["created_by"]=int(app.state.user_id)
        jd_id = app.state.jd_db_ops.create(result)
        
        if not jd_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save extracted JD"
            )
        
        logger.info(f"JD extracted and saved with ID: {jd_id}")
        return {
            "status": "success",
            "jd_fields": result,
            "jd_id": jd_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting JD fields: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract job description fields"
        )


@app.post("/upload/resume", tags=["Resume"])
async def upload_resume(
    jd_id: int = Form(...),
    resume: Optional[List[UploadFile]] = File(None)
):
    """Upload resume files (supports multiple files and ZIP)"""
    try:
        # Validate JD ID
        if jd_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid jd_id: must be positive integer"
            )
        
        jd_data = app.state.jd_db_ops.read(jd_id)
        if not jd_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job description not found for jd_id: {jd_id}"
            )
        
        # Validate files
        if not resume or not any(is_valid_upload(f) for f in resume):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid resume files provided"
            )
        
        saved_files = []
        extracted_files = []
        
        for file in resume:
            if not is_valid_upload(file):
                logger.warning(f"Skipping invalid file: {file.filename}")
                continue
            
            try:
                file_path = save_uploaded_file(RESUME_FOLDER, file)
                saved_files.append(file_path)
                
                # Extract ZIP files
                if file.filename.lower().endswith(".zip"):
                    zip_extract_path = os.path.splitext(file_path)[0]
                    extracted = extract_zip(file_path, zip_extract_path)
                    extracted_files.extend(extracted)
                    
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                # Continue with other files instead of failing completely
                continue
        
        all_uploaded = saved_files + extracted_files
        if not all_uploaded:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files were successfully uploaded"
            )
        
        # Store in app state
        app.state.resumes = all_uploaded
        app.state.jd_id_for_resume = jd_id
        
        logger.info(f"Uploaded {len(all_uploaded)} files for JD {jd_id}")
        return {
            "status": "success",
            "message": "Resumes uploaded successfully",
            "uploaded_files": saved_files,
            "extracted_from_zip": extracted_files,
            "total_files": len(all_uploaded)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading resumes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload resumes"
        )


@app.get("/extract_resume_fields", tags=["Resume"])
def extract_resume_fields():
    """Extract fields from uploaded resumes using AI"""
    try:
        if not app.state.resumes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No resumes uploaded. Please upload resumes first."
            )
        
        if not app.state.jd_id_for_resume:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No JD ID associated with resumes"
            )
        
        # Fetch JD data
        jd_result = app.state.jd_db_ops.read(app.state.jd_id_for_resume)
        if not jd_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job description not found for jd_id: {app.state.jd_id_for_resume}"
            )
        
        # Extract JD text
        extracted_json = jd_result.get("extracted_json", {})
        jd_fields_data = extracted_json.get("jd_fields", {})
        jd_text = jd_fields_data.get("jd_fields") or jd_fields_data
        
        if not jd_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="JD text not found in job description"
            )
        
        # Process resumes
        upload_list = app.state.resumes
        resume_data = []
        resume_conn = app.state.resume_db_ops
        app.state.resume_id_list = []
        
        graph = resume_agent_executor.create_executer(app.state.llm)
        
        for file_path in upload_list:
            try:
                logger.info(f"Processing resume: {file_path}")
                
                inputs = {
                    "jd_text": jd_text,
                    "resume_path": file_path,
                    "messages": []
                }
                
                output = graph.invoke(inputs)
                
                if not output or "resume_json" not in output:
                    logger.warning(f"Failed to extract data from {file_path}")
                    continue

                # Insert resume score
                if "resume_score" in output and output["resume_score"]:
                    score_dict = output["resume_score"].json( )
                    resume_json=output["resume_json"].dict()
                    resume_json["uploaded_by"]=int(app.state.user_id)
                    # resume_conn.insert_resume_score(resume_id, score_dict)
                # Insert resume into database
                resume_id = resume_conn.insert_resume(
                    app.state.jd_id_for_resume,
                    resume_json,
                    score_dict
                )
                if not resume_id:
                    logger.error(f"Failed to insert resume from {file_path}")
                    continue
                
                # # Insert resume score
                # if "resume_score" in output and output["resume_score"]:
                #     score_dict = output["resume_score"].dict() if hasattr(output["resume_score"], 'dict') else output["resume_score"]
                #     resume_conn.insert_resume_score(resume_id, score_dict)
                output_json=output["resume_json"].dict()
                output_json["resume_id"]=resume_id
                resume_data.append(output_json)
                app.state.resume_id_list.append(resume_id)
                logger.info(f"Processed resume ID: {resume_id}")
                
            except Exception as e:
                logger.error(f"Error processing resume {file_path}: {e}")
                # Continue with next resume
                continue
        
        # Clear state
        app.state.resumes = None
        
        if not resume_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process any resumes"
            )
        
        logger.info(f"Processed {len(resume_data)} resumes successfully")
        return {
            "status": "success",
            "message": f"Successfully extracted data from {len(resume_data)} resumes",
            "resume_data": resume_data,
            "jd_id": app.state.jd_id_for_resume,
            "resume_count": len(resume_data),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting resume fields: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract resume fields"
        )


@app.get("/fetch_resumes", tags=["Resume"])
def get_resumes(jd_id: Optional[int] = None, resume_id: Optional[int] = None):
    """Fetch resumes by JD ID or Resume ID"""
    try:
        if not jd_id and not resume_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please provide either jd_id or resume_id"
            )
        
        if jd_id and jd_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid jd_id: must be positive integer"
            )
        
        if resume_id and resume_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid resume_id: must be positive integer"
            )
        
        resume_data = []
        id_value = None
        
        if jd_id:
            data = app.state.resume_db_ops.get_resumes_by_jd(jd_id)
            if data:
                resume_data = data if isinstance(data, list) else [data]
            id_value = jd_id
        else:
            data = app.state.resume_db_ops.get_resume_by_id(resume_id)
            if data:
                resume_data = [data]
            id_value = resume_id
        
        if not resume_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No resumes found for the provided ID"
            )
        
        logger.info(f"Fetched {len(resume_data)} resumes")
        return {
            "status": "success",
            "message": "Resumes fetched successfully",
            "resume_data": resume_data,
            "id": id_value,
            "count": len(resume_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching resumes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch resumes"
        )


@app.get("/resume_score", tags=["Resume"])
def get_resume_score(resume_ids: Optional[str] = None):
    """
    Fetch resume scores by resume IDs
    
    Args:
        resume_ids: Comma-separated resume IDs (e.g., "1,2,3") or leave empty to use last extracted resumes
    
    Examples:
        /resume_score?resume_ids=1,2,3
        /resume_score (uses last extracted resumes)
    """
    try:
        scores = []
        resume_ids_to_fetch = []
        
        if resume_ids:
            # Parse comma-separated IDs
            try:
                id_list = [int(rid.strip()) for rid in resume_ids.split(',') if rid.strip()]
                print("id list: ",id_list)
                # Validate all IDs
                for rid in id_list:
                    if rid <= 0:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid resume_id: {rid}. Must be positive integer"
                        )
                resume_ids_to_fetch = id_list
                if not app.state.resume_id_list:
                    app.state.resume_id_list=resume_ids_to_fetch
                
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid resume_ids format. Use comma-separated integers (e.g., '1,2,3')"
                )
        elif app.state.resume_id_list:
            resume_ids_to_fetch = app.state.resume_id_list
            # resume_ids_to_fetch=[]
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please provide resume_ids parameter (e.g., ?resume_ids=1,2,3) or extract resumes first"
            )
        
        if not resume_ids_to_fetch:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid resume IDs provided"
            )
        
        for rid in app.state.resume_id_list:
            try:
                if rid in resume_ids_to_fetch:
                    score_data = app.state.resume_db_ops.fetch_resume_score(rid)
                    if score_data:
                        scores.append({
                            "resume_id": rid,
                            "score": score_data
                        })
                    else:
                        logger.warning(f"No score found for resume_id: {rid}")
                else:
                    logger.info(f"Removing resume data with ID: {rid}")
                    result=app.state.resume_db_ops.delete_resume(rid)
                    if result:
                        logger.info(f"Removed resume data with ID: {rid}")
                    
            except Exception as e:
                logger.error(f"Error fetching score for resume {rid}: {e}")
                # Continue with other IDs
                continue
        
        if not scores:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No scores found for the provided resume IDs"
            )
        app.state.resume_id_list.clear()
        logger.info(f"Fetched scores for {len(scores)} resumes")
        return {
            "status": "success",
            "message": "Resume scores fetched successfully",
            "scores": scores,
            "count": len(scores)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching resume scores: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch resume scores"
        )


@app.post("/resume_score/batch", tags=["Resume"])
async def get_resume_score_batch(request: Request):
    """
    Fetch resume scores by resume IDs (POST method for large lists)
    
    Request Body:
        {
            "resume_ids": [1, 2, 3, 4, 5]
        }
    """
    try:
        data = await request.json()
        
        if not data or "resume_ids" not in data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request body must contain 'resume_ids' array"
            )
        
        resume_ids_to_fetch = data["resume_ids"]
        
        if not isinstance(resume_ids_to_fetch, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="'resume_ids' must be an array of integers"
            )
        
        if not resume_ids_to_fetch:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="'resume_ids' array cannot be empty"
            )
        
        # Validate all IDs
        for rid in resume_ids_to_fetch:
            if not isinstance(rid, int) or rid <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid resume_id: {rid}. Must be positive integer"
                )
        
        scores = []
        
        for rid in resume_ids_to_fetch:
            try:
                score_data = app.state.resume_db_ops.fetch_resume_score(rid)
                
                if score_data:
                    scores.append({
                        "resume_id": rid,
                        "score": score_data
                    })
                else:
                    logger.warning(f"No score found for resume_id: {rid}")
                    
            except Exception as e:
                logger.error(f"Error fetching score for resume {rid}: {e}")
                # Continue with other IDs
                continue
        
        if not scores:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No scores found for the provided resume IDs"
            )
        
        logger.info(f"Fetched scores for {len(scores)} resumes")
        return {
            "status": "success",
            "message": "Resume scores fetched successfully",
            "scores": scores,
            "count": len(scores),
            "requested_count": len(resume_ids_to_fetch)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching resume scores: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch resume scores"
        )
