import os
import sqlite3
import pandas as pd

DATABASE_URL = os.getenv("DATABASE_URL", "")
DATABASE_PATH = os.getenv("DATABASE_PATH", "./proxy.db")

class DBConnectionWrapper:
    def __init__(self, conn, is_pg=False):
        self.conn = conn
        self.is_pg = is_pg

    def execute(self, query, params=()):
        if self.is_pg:
            query = query.replace("?", "%s")
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return cursor

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

def get_db_connection():
    if DATABASE_URL.startswith("postgres"):
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.DictCursor)
        return DBConnectionWrapper(conn, is_pg=True)
    else:
        conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

def init_db() -> None:
    """Create the required database tables if they don't exist."""
    conn = get_db_connection()
    try:
        is_pg = DATABASE_URL.startswith("postgres")
        
        id_type = "TEXT PRIMARY KEY"
        
        # Users table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        # API Keys table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                key_hash TEXT NOT NULL,
                prefix TEXT NOT NULL,
                name TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        # Requests table
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS requests (
                id {id_type},
                timestamp TEXT NOT NULL,
                user_id TEXT,
                prompt_text TEXT NOT NULL,
                system_prompt TEXT,
                score_heuristic REAL,
                score_classifier REAL,
                score_embedding REAL,
                score_judge REAL,
                final_score REAL NOT NULL,
                action_taken TEXT NOT NULL,
                triggered_layers TEXT NOT NULL,
                matched_patterns TEXT,
                judge_reason TEXT,
                model TEXT,
                processing_ms REAL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        
        if is_pg:
            try:
                # Fix for previously created SERIAL column in Postgres
                conn.execute("ALTER TABLE requests ALTER COLUMN id TYPE TEXT")
            except Exception:
                pass
        
        if not is_pg:
            try:
                conn.execute("ALTER TABLE requests ADD COLUMN user_id TEXT")
            except Exception:
                pass
                
        conn.commit()
    finally:
        conn.close()

def query_db_df(query: str, params: tuple = (), conn=None) -> pd.DataFrame:
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True
        
    if DATABASE_URL.startswith("postgres"):
        query = query.replace("?", "%s")
        
    try:
        # Pandas requires the raw DBAPI connection object, not our wrapper
        raw_conn = conn.conn if hasattr(conn, "conn") else conn
        return pd.read_sql_query(query, raw_conn, params=params)
    except Exception as e:
        print(f"DB Error: {e}")
        return pd.DataFrame()
    finally:
        if close_conn:
            conn.close()
