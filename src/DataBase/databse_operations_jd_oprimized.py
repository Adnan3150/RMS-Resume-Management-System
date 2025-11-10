import os
import json
import psycopg2
from psycopg2.extras import Json
from psycopg2 import pool
from dotenv import load_dotenv
from pathlib import Path
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=ROOT_DIR / ".env")

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}


class JobDescriptionCRUD:
    def __init__(self, config, pool_size=5, max_overflow=10):
        """Initialize with connection pooling for better performance"""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=pool_size + max_overflow,
                **config
            )
            logger.info("Database connection pool created successfully")
        except psycopg2.Error as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise

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

    def create(self, jd_json):
        """Insert new Job Description with proper transaction handling"""
        if not jd_json or "jd_fields" not in jd_json:
            raise ValueError("Invalid jd_json: missing 'jd_fields'")

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    jd_fields = jd_json["jd_fields"]
                    search_queries = jd_json.get("search_strings", {}).get("naukri_search_strings")
                    
                    logger.info(f"Inserting JD: {jd_fields.get('job_title', 'N/A')}")
                    
                    insert_query = """
                    INSERT INTO job_descriptions (
                        jd_title, company_name, location, experience,
                        must_have_skills, nice_to_have_skills,
                        raw_text, extracted_json, naukri_search_query, created_by, employment_type
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING jd_id;
                    """

                    values = (
                        jd_fields.get("job_title"),
                        jd_fields.get("company_name"),
                        jd_fields.get("location"),
                        jd_fields.get("experience"),
                        jd_fields.get("must_have_skills"),
                        jd_fields.get("nice_to_have_skills"),
                        None, #need to insert extracted text
                        Json(jd_json),
                        search_queries,
                        jd_json.get("created_by"),
                        jd_fields.get("employment_type")
                    )

                    cur.execute(insert_query, values)
                    jd_id = cur.fetchone()[0]
                    logger.info(f"Inserted Job Description with ID: {jd_id}")
                    return jd_id
                    
        except psycopg2.IntegrityError as e:
            logger.error(f"Integrity error while inserting JD: {e}")
            raise ValueError(f"Database integrity error: {e}")
        except psycopg2.Error as e:
            logger.error(f"Database error while inserting JD: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while inserting JD: {e}")
            raise

    def read(self, jd_id):
        """Fetch a single job description and return as dict"""
        if not isinstance(jd_id, int) or jd_id <= 0:
            raise ValueError("Invalid jd_id: must be a positive integer")

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM job_descriptions WHERE jd_id = %s;",
                        (jd_id,)
                    )
                    result = cur.fetchone()

                    if not result:
                        logger.warning(f"No record found for jd_id = {jd_id}")
                        return None

                    columns = [desc[0] for desc in cur.description]
                    jd_dict = dict(zip(columns, result))
                    logger.info(f"✅ Fetched JD with ID: {jd_id}")
                    return jd_dict
                    
        except psycopg2.Error as e:
            logger.error(f"Database error while reading JD {jd_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while reading JD {jd_id}: {e}")
            raise

    def update(self, jd_id, field_name, new_value):
        """Update a specific field for a given jd_id"""
        if not isinstance(jd_id, int) or jd_id <= 0:
            raise ValueError("Invalid jd_id: must be a positive integer")
        
        if not field_name or not isinstance(field_name, str):
            raise ValueError("Invalid field_name")

        # Whitelist allowed fields to prevent SQL injection
        allowed_fields = {
            'jd_title', 'company_name', 'location', 'experience',
            'must_have_skills', 'nice_to_have_skills', 'raw_text',
            'extracted_json', 'naukri_search_query', 'employment_type'
        }
        
        if field_name not in allowed_fields:
            raise ValueError(f"Field '{field_name}' is not allowed for updates")

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = f"UPDATE job_descriptions SET {field_name} = %s WHERE jd_id = %s;"
                    cur.execute(query, (new_value, jd_id))
                    
                    if cur.rowcount == 0:
                        logger.warning(f"No rows updated for jd_id = {jd_id}")
                        return False
                    
                    logger.info(f"✅ Updated {field_name} for Job ID {jd_id}")
                    return True
                    
        except psycopg2.Error as e:
            logger.error(f"Database error while updating JD {jd_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while updating JD {jd_id}: {e}")
            raise

    def delete(self, jd_id):
        """Delete a job description"""
        if not isinstance(jd_id, int) or jd_id <= 0:
            raise ValueError("Invalid jd_id: must be a positive integer")

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM job_descriptions WHERE jd_id = %s;",
                        (jd_id,)
                    )
                    
                    if cur.rowcount == 0:
                        logger.warning(f"No rows deleted for jd_id = {jd_id}")
                        return False
                    
                    logger.info(f"🗑️ Deleted Job Description ID {jd_id}")
                    return True
                    
        except psycopg2.Error as e:
            logger.error(f"Database error while deleting JD {jd_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while deleting JD {jd_id}: {e}")
            raise

    def close(self):
        """Close all connections in the pool"""
        try:
            self.connection_pool.closeall()
            logger.info("Database connection pool closed")
        except Exception as e:
            logger.error(f"Error closing connection pool: {e}")


def initialize_db():
    """Initialize database connection with error handling"""
    try:
        crud = JobDescriptionCRUD(DB_CONFIG)
        return crud
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise