import os
import json
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  

load_dotenv(dotenv_path=ROOT_DIR / ".env")  

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# ----------------------------
# CRUD Class Definition
# ----------------------------
class JobDescriptionCRUD:
    def __init__(self, config):
        self.conn = psycopg2.connect(**config)
        self.cur = self.conn.cursor()

    def create(self, jd_json):
        """Insert new Job Description"""
        try:
            jd_fields = jd_json["jd_fields"]
            search_queries = jd_json["search_strings"]["naukri_search_strings"]
            print("jd_fields",jd_fields)
            insert_query = """
            INSERT INTO job_descriptions (
                jd_title, company_name, location, experience,
                must_have_skills, nice_to_have_skills,
                raw_text, extracted_json, naukri_search_query,employment_type
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,%s)
            RETURNING jd_id;
            """

            company_name = jd_fields.get("company_name")
            values = (
                jd_fields.get("job_title"),
                company_name,
                jd_fields.get("location"),
                jd_fields.get("experience"),
                jd_fields.get("must_have_skills"),
                jd_fields.get("nice_to_have_skills"),
                None,  # raw_text optional
                Json(jd_json),  # Store the full structured JSON
                search_queries,
                jd_fields.get("employment_type")
            )

            self.cur.execute(insert_query, values)
            jd_id = self.cur.fetchone()[0]
            self.conn.commit()
            print(f"✅ Inserted Job Description with ID: {jd_id}")
            return jd_id
        except Exception as e:
            print(f"❌ Database error: {e}")
            self.conn.rollback()  # ✅ reset failed transaction
            return None

    def read(self, jd_id):
        """Fetch a single job description and return as JSON object with column names"""
        try:
            self.cur.execute("SELECT * FROM job_descriptions WHERE jd_id = %s;", (jd_id,))
            result = self.cur.fetchone()

            if not result:
                print(f"❌ No record found for jd_id = {jd_id}")
                return None
            # Get column names from cursor description
            columns = [desc[0] for desc in self.cur.description]
            # Map columns to values
            jd_dict = dict(zip(columns, result))
            # Convert to JSON serializable object (dict)
            return jd_dict
        except Exception as e:
            print(f"❌ Database error: {e}")
            self.conn.rollback()  # ✅ reset failed transaction
            return None

    def update(self, jd_id, field_name, new_value):
        """Update a specific field for a given jd_id"""
        try:
            query = f"UPDATE job_descriptions SET {field_name} = %s WHERE jd_id = %s;"
            self.cur.execute(query, (new_value, jd_id))
            self.conn.commit()
            print(f"✅ Updated {field_name} for Job ID {jd_id}")
        except Exception as e:
            print(f"❌ Database error: {e}")
            self.conn.rollback()  # ✅ reset failed transaction
            return None

    def delete(self, jd_id):
        """Delete a job description"""
        self.cur.execute("DELETE FROM job_descriptions WHERE jd_id = %s;", (jd_id,))
        self.conn.commit()
        print(f"🗑️ Deleted Job Description ID {jd_id}")

    def close(self):
        self.cur.close()
        self.conn.close()

def initialize_db():
    crud = JobDescriptionCRUD(DB_CONFIG)
    return crud

    