import psycopg2
import json
import logging
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RolesUsersDatabase:
    def __init__(self, conn):
        """Initialize with connection pool"""
        if not hasattr(conn, 'connection_pool'):
            raise ValueError("Invalid connection object: missing connection_pool attribute")
        self.connection_pool = conn.connection_pool
        logger.info("RolesUsersDatabase initialized with connection pool")

    @contextmanager
    def get_connection(self):
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

    # ----------------------------------------------------
    # ROLE OPERATIONS
    # ----------------------------------------------------
    def create_roles_table(self):
        """Create roles table if not exists"""
        query = """
        CREATE TABLE IF NOT EXISTS roles (
            role_id SERIAL PRIMARY KEY,
            role_name VARCHAR(100) UNIQUE NOT NULL,
            role_description TEXT
        );
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                logger.info("✅ Roles table created/verified successfully")

    def insert_role(self, role_name: str, description: str):
        """Insert a new role"""
        query = """
        INSERT INTO roles (role_name, role_description)
        VALUES (%s, %s)
        ON CONFLICT (role_name) DO NOTHING
        RETURNING role_id;
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (role_name, description))
                result = cur.fetchone()
                if result:
                    logger.info(f"✅ Role '{role_name}' inserted successfully (ID: {result['role_id']})")
                    return result['role_id']
                else:
                    logger.warning(f"⚠️ Role '{role_name}' already exists.")
                    return None

    def fetch_all_roles(self):
        """Fetch all roles"""
        query = "SELECT * FROM roles ORDER BY role_id;"
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                roles = cur.fetchall()
                logger.info(f"✅ Retrieved {len(roles)} roles")
                return [dict(r) for r in roles]

    # ----------------------------------------------------
    # USER OPERATIONS
    # ----------------------------------------------------
    def create_users_table(self):
        """Create users table with foreign key to roles"""
        query = """
        CREATE TABLE IF NOT EXISTS users (
            employee_id SERIAL PRIMARY KEY,
            username VARCHAR(150) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name VARCHAR(255),
            role_id INT NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_role FOREIGN KEY (role_id)
                REFERENCES roles (role_id)
                ON DELETE RESTRICT
        );
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                logger.info("✅ Users table created/verified successfully")

    def insert_user(self, employee_id, username, email, password_hash, role_id):
        """Insert user if role exists"""
        query_check = "SELECT 1 FROM roles WHERE role_id = %s;"
        query_insert = """
        INSERT INTO users (employee_id, username, email, password_hash, role_id)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING employee_id;
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query_check, (role_id,))
                if not cur.fetchone():
                    logger.error(f"❌ Role ID {role_id} not found — transaction aborted.")
                    raise ValueError(f"Role ID {role_id} does not exist in roles table.")
                cur.execute(query_insert, (employee_id, username, email, password_hash, role_id))
                user = cur.fetchone()
                logger.info(f"✅ User '{username}' inserted with Employee ID: {user['employee_id']}")
                return user['employee_id']

    def fetch_all_users(self):
        """Fetch all users with role names"""
        query = """
        SELECT u.employee_id, u.username, u.email, u.full_name, r.role_name, u.created_at
        FROM users u
        JOIN roles r ON u.role_id = r.role_id
        ORDER BY u.employee_id;
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                users = cur.fetchall()
                logger.info(f"✅ Retrieved {len(users)} users")
                return [dict(u) for u in users]

    def fetch_user_by_id(self, employee_id: int):
        """Fetch single user"""
        query = """
        SELECT u.employee_id, u.username, u.email, r.role_name, u.created_at, u.password_hash
        FROM users u
        JOIN roles r ON u.role_id = r.role_id
        WHERE u.employee_id = %s;
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (employee_id,))
                user = cur.fetchone()
                if not user:
                    logger.warning(f"⚠️ No user found with employee_id {employee_id}")
                    return None
                logger.info(f"✅ Retrieved user with employee_id: {employee_id}")
                return dict(user)

    def delete_user(self, employee_id: int):
        """Delete user by ID"""
        query = "DELETE FROM users WHERE employee_id = %s;"
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (employee_id,))
                if cur.rowcount == 0:
                    logger.warning(f"⚠️ No user found with employee_id {employee_id}")
                    return False
                logger.info(f"🗑️ Deleted user with employee_id {employee_id}")
                return True

    def close(self):
        """Close database connections"""
        try:
            if hasattr(self.conn_pool, 'close'):
                self.conn_pool.close()
            logger.info("RolesUsersDatabase connections closed")
        except Exception as e:
            logger.error(f"Error closing RolesUsersDatabase: {e}")


def initialize_db(conn):
    """Initialize ResumeDatabase with existing connection"""
    try:
        return RolesUsersDatabase(conn)
    except Exception as e:
        logger.error(f"Failed to initialize RolesUsersDatabase: {e}")
        raise