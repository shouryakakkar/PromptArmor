import os
import sqlite3
import pandas as pd

DATABASE_URL = os.getenv("DATABASE_URL", "")
DATABASE_PATH = os.getenv("DATABASE_PATH", "./proxy.db")

def get_db_connection():
    if DATABASE_URL.startswith("postgres"):
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.DictCursor)
        # Add a convenience execute method to act like sqlite3.Connection.execute
        if not hasattr(conn, "_execute_bound"):
            original_commit = conn.commit
            def execute(query, params=()):
                query = query.replace("?", "%s")
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor
            conn.execute = execute
        return conn
    else:
        conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

def init_db() -> None:
    """Create the required database tables if they don't exist."""
    conn = get_db_connection()
    try:
        is_pg = DATABASE_URL.startswith("postgres")
        
        id_type = "SERIAL PRIMARY KEY" if is_pg else "INTEGER PRIMARY KEY AUTOINCREMENT"
        
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
        return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        print(f"DB Error: {e}")
        return pd.DataFrame()
    finally:
        if close_conn:
            conn.close()
