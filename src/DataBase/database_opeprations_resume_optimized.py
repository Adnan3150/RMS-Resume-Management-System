import os
import json
import psycopg2
import traceback
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv
from pathlib import Path
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Environment
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=ROOT_DIR / ".env")

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}


class ResumeDatabase:
    def __init__(self, conn):
        """Initialize with existing connection pool from JobDescriptionCRUD"""
        if not hasattr(conn, 'connection_pool'):
            raise ValueError("Invalid connection object: missing connection_pool attribute")
        
        self.connection_pool = conn.connection_pool
        logger.info("ResumeDatabase initialized with connection pool")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections with automatic cleanup"""
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
            conn.commit()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def create_table(self):
        """Ensure resumes table exists"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    create_table_query = """
                    CREATE TABLE IF NOT EXISTS resumes (
                        resume_id SERIAL PRIMARY KEY,
                        jd_id INTEGER NOT NULL,
                        candidate_name VARCHAR(255),
                        email VARCHAR(255),
                        phone_number VARCHAR(50),
                        linkedin TEXT,
                        github TEXT,
                        total_experience VARCHAR(100),
                        employment_details JSONB,
                        current_company VARCHAR(255),
                        current_designation VARCHAR(255),
                        skills TEXT,
                        education TEXT,
                        certifications TEXT,
                        projects JSONB,
                        location VARCHAR(255),
                        preferred_location VARCHAR(255),
                        notice_period VARCHAR(100),
                        expected_salary VARCHAR(100),
                        languages TEXT,
                        summary TEXT,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW(),
                        CONSTRAINT fk_job_description
                            FOREIGN KEY (jd_id)
                            REFERENCES job_descriptions (jd_id)
                            ON DELETE CASCADE
                    );
                    """
                    cur.execute(create_table_query)
                    logger.info("✅ Resumes table created/verified successfully")
        except psycopg2.Error as e:
            logger.error(f"Error creating resumes table: {e}")
            raise

    def insert_resume(self, jd_id: int, data, score_json):
        """Insert resume data into the resumes table with proper validation"""
        if not isinstance(jd_id, int) or jd_id <= 0:
            raise ValueError("Invalid jd_id: must be a positive integer")

        if not data:
            raise ValueError("Resume data cannot be empty")

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    insert_query = """
                    INSERT INTO resumes (
                        jd_id, uploaded_by, candidate_name, email, phone_number, linkedin, github,
                        total_experience, employment_details, current_company, current_designation,
                        skills, education, certifications, projects,
                        location, preferred_location, notice_period, expected_salary,
                        languages, summary
                    )
                    VALUES (
                        %(jd_id)s, %(uploaded_by)s, %(candidate_name)s, %(email)s, %(phone_number)s, %(linkedin)s, %(github)s,
                        %(total_experience)s, %(employment_details)s, %(current_company)s, %(current_designation)s,
                        %(skills)s, %(education)s, %(certifications)s, %(projects)s,
                        %(location)s, %(preferred_location)s, %(notice_period)s, %(expected_salary)s,
                        %(languages)s, %(summary)s
                    ) RETURNING resume_id;
                    """

                    # Convert data to dictionary if needed
                    if not isinstance(data, dict):
                        if hasattr(data, "model_dump"):
                            safe_data = data.model_dump()
                        elif hasattr(data, "dict"):
                            safe_data = data.dict()
                        else:
                            raise TypeError(f"Unsupported data type: {type(data)}")
                    else:
                        safe_data = data.copy()

                    # Add jd_id
                    safe_data["jd_id"] = jd_id

                    # Convert JSON/array-like fields properly
                    json_fields = ["employment_details", "projects"]
                    for field in json_fields:
                        if field in safe_data and safe_data[field] is not None:
                            if isinstance(safe_data[field], (list, dict)):
                                safe_data[field] = json.dumps(safe_data[field])
                            elif not isinstance(safe_data[field], str):
                                safe_data[field] = json.dumps([safe_data[field]])
                    cur.execute("SET LOCAL resume.score_json = %s;", (score_json,))
                    # Execute insertion
                    cur.execute(insert_query, safe_data)
                    result = cur.fetchone()
                    
                    if not result:
                        raise psycopg2.Error("Failed to retrieve resume_id after insertion")
                    
                    resume_id = result["resume_id"]
                    logger.info(f" Resume inserted successfully with ID: {resume_id}")
                    return resume_id

        except psycopg2.IntegrityError as e:
            logger.error(f"Integrity error inserting resume: {e}")
            raise ValueError(f"Database integrity error: {e}")
        except psycopg2.Error as e:
            logger.error(f"Database error inserting resume: {e}")
            if hasattr(e, 'pgcode'):
                logger.error(f"PGCODE: {e.pgcode}")
            if hasattr(e, 'pgerror'):
                logger.error(f"PGERROR: {e.pgerror}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error inserting resume: {e}")
            traceback.print_exc()
            raise

    def get_resume_by_id(self, resume_id: int):
        """Fetch single resume by resume_id"""
        if not isinstance(resume_id, int) or resume_id <= 0:
            raise ValueError("Invalid resume_id: must be a positive integer")

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT * FROM resumes WHERE resume_id = %s;",
                        (resume_id,)
                    )
                    result = cur.fetchone()
                    
                    if not result:
                        logger.warning(f"No resume found for resume_id = {resume_id}")
                        return None
                    
                    logger.info(f"✅ Fetched resume with ID: {resume_id}")
                    return dict(result)
                    
        except psycopg2.Error as e:
            logger.error(f"Database error fetching resume {resume_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching resume {resume_id}: {e}")
            raise

    def get_resumes_by_jd(self, jd_id: int):
        """Fetch all resumes linked to a specific jd_id"""
        if not isinstance(jd_id, int) or jd_id <= 0:
            raise ValueError("Invalid jd_id: must be a positive integer")

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT * FROM resumes WHERE jd_id = %s ORDER BY created_at DESC;",
                        (jd_id,)
                    )
                    results = cur.fetchall()
                    
                    if not results:
                        logger.warning(f"No resumes found for jd_id = {jd_id}")
                        return []
                    
                    logger.info(f"✅ Fetched {len(results)} resumes for JD ID: {jd_id}")
                    return [dict(row) for row in results]
                    
        except psycopg2.Error as e:
            logger.error(f"Database error fetching resumes for JD {jd_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching resumes for JD {jd_id}: {e}")
            raise

    def update_resume(self, resume_id: int, update_data: dict):
        """Update resume fields dynamically"""
        if not isinstance(resume_id, int) or resume_id <= 0:
            raise ValueError("Invalid resume_id: must be a positive integer")

        if not update_data or not isinstance(update_data, dict):
            raise ValueError("Invalid update_data: must be a non-empty dictionary")

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    set_clauses = []
                    values = []
                    
                    for key, value in update_data.items():
                        set_clauses.append(f"{key} = %s")
                        if isinstance(value, (list, dict)):
                            value = json.dumps(value)
                        values.append(value)
                    
                    values.append(resume_id)

                    update_query = f"""
                    UPDATE resumes
                    SET {', '.join(set_clauses)}, updated_at = NOW()
                    WHERE resume_id = %s;
                    """

                    cur.execute(update_query, values)
                    
                    if cur.rowcount == 0:
                        logger.warning(f"No rows updated for resume_id = {resume_id}")
                        return False
                    
                    logger.info(f"✅ Resume ID {resume_id} updated successfully")
                    return True
                    
        except psycopg2.Error as e:
            logger.error(f"Database error updating resume {resume_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating resume {resume_id}: {e}")
            raise

    def delete_resume(self, resume_id: int):
        """Delete a resume by ID"""
        if not isinstance(resume_id, int) or resume_id <= 0:
            raise ValueError("Invalid resume_id: must be a positive integer")

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM resumes WHERE resume_id = %s;",
                        (resume_id,)
                    )
                    
                    if cur.rowcount == 0:
                        logger.warning(f"No rows deleted for resume_id = {resume_id}")
                        return False
                    
                    logger.info(f"🗑️ Resume ID {resume_id} deleted successfully")
                    return True
                    
        except psycopg2.Error as e:
            logger.error(f"Database error deleting resume {resume_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting resume {resume_id}: {e}")
            raise

    def insert_resume_score(self, resume_id: int, resume_score: dict):
        """Insert resume score with proper validation"""
        if not isinstance(resume_id, int) or resume_id <= 0:
            raise ValueError("Invalid resume_id: must be a positive integer")

        if not resume_score or not isinstance(resume_score, dict):
            raise ValueError("Invalid resume_score: must be a non-empty dictionary")

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    insert_query = """
                    INSERT INTO resume_scores (resume_id, resume_score)
                    VALUES (%s, %s)
                    RETURNING resume_id;
                    """

                    values = (resume_id, Json(resume_score))
                    cur.execute(insert_query, values)
                    result = cur.fetchone()
                    
                    if not result:
                        raise psycopg2.Error("Failed to retrieve resume_id after score insertion")
                    
                    logger.info(f"✅ Resume score inserted for ID: {resume_id}")
                    return result["resume_id"]

        except psycopg2.IntegrityError as e:
            logger.error(f"Integrity error inserting resume score: {e}")
            raise ValueError(f"Database integrity error: {e}")
        except psycopg2.Error as e:
            logger.error(f"Database error inserting resume score: {e}")
            if hasattr(e, 'pgcode'):
                logger.error(f"PGCODE: {e.pgcode}")
            if hasattr(e, 'pgerror'):
                logger.error(f"PGERROR: {e.pgerror}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error inserting resume score: {e}")
            traceback.print_exc()
            raise

    def fetch_resume_score(self, resume_id: int):
        """Fetch resume score by resume_id"""
        if not isinstance(resume_id, int) or resume_id <= 0:
            raise ValueError("Invalid resume_id: must be a positive integer")

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT * FROM resume_scores WHERE resume_id = %s;",
                        (resume_id,)
                    )
                    result = cur.fetchone()
                    
                    if not result:
                        logger.warning(f"No score found for resume_id = {resume_id}")
                        return None
                    
                    logger.info(f"✅ Fetched score for resume ID: {resume_id}")
                    return dict(result)
                    
        except psycopg2.Error as e:
            logger.error(f"Database error fetching score for resume {resume_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching score for resume {resume_id}: {e}")
            raise

    def close(self):
        """Close database connections"""
        try:
            if hasattr(self.conn_pool, 'close'):
                self.conn_pool.close()
            logger.info("ResumeDatabase connections closed")
        except Exception as e:
            logger.error(f"Error closing ResumeDatabase: {e}")


def initialize_db(conn):
    """Initialize ResumeDatabase with existing connection"""
    try:
        return ResumeDatabase(conn)
    except Exception as e:
        logger.error(f"Failed to initialize ResumeDatabase: {e}")
        raise