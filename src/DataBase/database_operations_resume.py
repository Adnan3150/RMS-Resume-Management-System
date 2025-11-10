import os
import json
import psycopg2
import traceback
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv
from pathlib import Path

# ------------------ Load Environment ------------------
ROOT_DIR = Path(__file__).resolve().parents[2]  # project root
load_dotenv(dotenv_path=ROOT_DIR / ".env")

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}


# ------------------ Database Class ------------------
class ResumeDatabase:
    def __init__(self,conn):
        # self.conn = psycopg2.connect(**DB_CONFIG)
        self.conn=conn.conn
        self.cur = self.conn.cursor(cursor_factory=RealDictCursor)

    def create_table(self):
        """Ensure resumes table exists"""
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
            current_company VARCHAR(255),
            current_designation VARCHAR(255),
            skills TEXT,
            education TEXT,
            certifications TEXT,
            projects TEXT,
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
        self.cur.execute(create_table_query)
        self.conn.commit()

    # ------------------ INSERT ------------------
    # def insert_resume(self, jd_id: int, data: dict):
    #     """Insert resume data"""
    #     try:
    #         insert_query = """
    #         INSERT INTO resumes (
    #             jd_id, candidate_name, email, phone_number, linkedin, github,
    #             total_experience, current_company, current_designation,
    #             skills, education, certifications, projects,
    #             location, preferred_location, notice_period, expected_salary,
    #             languages, summary
    #         )
    #         VALUES (
    #             %(jd_id)s, %(candidate_name)s, %(email)s, %(phone_number)s, %(linkedin)s, %(github)s,
    #             %(total_experience)s, %(current_company)s, %(current_designation)s,
    #             %(skills)s, %(education)s, %(certifications)s, %(projects)s,
    #             %(location)s, %(preferred_location)s, %(notice_period)s, %(expected_salary)s,
    #             %(languages)s, %(summary)s
    #         ) RETURNING resume_id;
    #         """

    #         # Convert list fields to JSON/text
    #         data = data.dict().copy()
    #         # list_fields = ["skills", "education", "certifications", "projects", "languages"]
    #         # for field in list_fields:
    #         #     if isinstance(data.get(field), list):
    #         #         data[field] = json.dumps(data[field])

    #         data["jd_id"] = jd_id

    #         self.cur.execute(insert_query, data)
    #         resume_id = self.cur.fetchone()["resume_id"]
    #         self.conn.commit()
    #         print(f"✅ Resume inserted successfully with ID: {resume_id}")
    #         return resume_id
    #     except Exception as e:
    #         print(f"❌ Database error: {e}")
    #         self.conn.rollback()  # ✅ reset failed transaction
    #         return None
    def insert_resume(self, jd_id: int, data):
        """Insert resume data into the resumes table"""
        try:
            insert_query = """
            INSERT INTO resumes (
                jd_id, candidate_name, email, phone_number, linkedin, github,
                total_experience, employment_details, current_company, current_designation,
                skills, education, certifications, projects,
                location, preferred_location, notice_period, expected_salary,
                languages, summary
            )
            VALUES (
                %(jd_id)s, %(candidate_name)s, %(email)s, %(phone_number)s, %(linkedin)s, %(github)s,
                %(total_experience)s, %(employment_details)s, %(current_company)s, %(current_designation)s,
                %(skills)s, %(education)s, %(certifications)s, %(projects)s,
                %(location)s, %(preferred_location)s, %(notice_period)s, %(expected_salary)s,
                %(languages)s, %(summary)s
            ) RETURNING resume_id;
            """

            # ✅ Ensure `data` is a plain dictionary
            if not isinstance(data, dict):
                if hasattr(data, "model_dump"):  # Pydantic v2
                    safe_data = data.model_dump()
                elif hasattr(data, "dict"):      # Pydantic v1
                    safe_data = data.dict()
                else:
                    raise TypeError("Unsupported data type for resume insertion")
            else:
                safe_data = data.copy()

            # ✅ Add jd_id
            safe_data["jd_id"] = jd_id

            # ✅ Convert JSON/array-like fields properly
            json_fields = [
                "employment_details", "projects"
                
            ]

            for field in json_fields:
                if field in safe_data and safe_data[field] is not None:
                    if isinstance(safe_data[field], (list, dict)):
                        safe_data[field] = json.dumps(safe_data[field])
                    else:
                        safe_data[field] = json.dumps([safe_data[field]])  # handle str fallback
            # ✅ Execute insertion
            self.cur.execute(insert_query, safe_data)
            resume_id = self.cur.fetchone()["resume_id"]
            self.conn.commit()
            print(f"✅ Resume inserted successfully with ID: {resume_id}")
            return resume_id

        except Exception as e:
            print("❌ Database error:", e)
            if isinstance(e, psycopg2.Error):
                print("🧩 PGCODE:", e.pgcode)
                print("📜 PGERROR:", e.pgerror)
            traceback.print_exc()
            self.conn.rollback()
            return None

    # ------------------ READ ------------------
    def get_resume_by_id(self, resume_id: int):
        """Fetch single resume by resume_id"""
        try:
            self.cur.execute("SELECT * FROM resumes WHERE resume_id = %s;", (resume_id,))
            result = self.cur.fetchone()
            return json.dumps(result, indent=4, default=str) if result else None
        except Exception as e:
            print(f"❌ Database error: {e}")
            self.conn.rollback()  # ✅ reset failed transaction
            return None
    def get_resumes_by_jd(self, jd_id: int):
        try:
            """Fetch all resumes linked to a specific jd_id"""
            self.cur.execute("SELECT * FROM resumes WHERE jd_id = %s;", (jd_id,))
            results = self.cur.fetchall()
            return json.dumps(results, indent=4, default=str) if results else None
        except Exception as e:
            print(f"❌ Database error: {e}")
            self.conn.rollback()  # ✅ reset failed transaction
            return None

    # ------------------ UPDATE ------------------
    def update_resume(self, resume_id: int, update_data: dict):
        """Update resume fields dynamically"""
        set_clauses = []
        values = []
        for key, value in update_data.items():
            set_clauses.append(f"{key} = %s")
            if isinstance(value, list):
                value = json.dumps(value)
            values.append(value)
        values.append(resume_id)

        update_query = f"""
        UPDATE resumes
        SET {', '.join(set_clauses)}, updated_at = NOW()
        WHERE resume_id = %s;
        """

        self.cur.execute(update_query, values)
        self.conn.commit()
        print(f"✅ Resume ID {resume_id} updated successfully.")

    # ------------------ DELETE ------------------
    def delete_resume(self, resume_id: int):
        """Delete a resume by ID"""
        self.cur.execute("DELETE FROM resumes WHERE resume_id = %s;", (resume_id,))
        self.conn.commit()
        print(f"🗑️ Resume ID {resume_id} deleted successfully.")


    def insert_resume_score(self, resume_id:int, resume_score:dict):
        try:
            insert_query = """
                        INSERT INTO resume_scores (
                            resume_id, score
                        )
                        VALUES (%s, %s)
                        RETURNING resume_id;
                        """

            values = (resume_id, Json(resume_score))

            self.cur.execute(insert_query, values)
            resume_id=self.cur.fetchone()
            self.conn.commit()
            print(f"✅ Resume inserted successfully with ID: {resume_id}")
            return resume_id
            
        except Exception as e:
            print("❌ Database error:", e)
            if isinstance(e, psycopg2.Error):
                print("🧩 PGCODE:", e.pgcode)
                print("📜 PGERROR:", e.pgerror)
            traceback.print_exc()
            self.conn.rollback()
            return None

    def fetch_resume_score(self, resume_id:int):
        try:
            self.cur.execute("SELECT * FROM resume_scores WHERE resume_id = %s;", (resume_id,))
            result = self.cur.fetchone()
            return json.dumps(result, indent=4, default=str) if result else None
        except Exception as e:
            print(f"❌ Database error: {e}")
            self.conn.rollback()  # ✅ reset failed transaction
            return None

    # ------------------ CLOSE ------------------
    def close(self):
        self.cur.close()
        self.conn.close()

def initialize_db(conn):
    return ResumeDatabase(conn)
