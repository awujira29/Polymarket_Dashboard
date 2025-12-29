from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from models import Base
from config import DATABASE_URL
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create engine
engine = create_engine(DATABASE_URL, echo=False)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize the database - create all tables"""
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    _ensure_sqlite_schema()
    logger.info("Database tables created successfully!")

def _ensure_sqlite_schema():
    """Apply lightweight SQLite migrations for newly added columns."""
    if not DATABASE_URL.startswith("sqlite"):
        return

    required_columns = {
        "markets": {
            "subcategory": "VARCHAR",
            "event_title": "TEXT",
            "event_tags": "TEXT",
            "outcomes": "TEXT"
        }
    }

    with engine.begin() as conn:
        for table_name, columns in required_columns.items():
            existing = {
                row[1] for row in conn.execute(text(f"PRAGMA table_info({table_name});"))
            }
            for column_name, column_type in columns.items():
                if column_name not in existing:
                    logger.info(
                        "Adding missing column %s.%s",
                        table_name,
                        column_name
                    )
                    conn.execute(
                        text(
                            f"ALTER TABLE {table_name} "
                            f"ADD COLUMN {column_name} {column_type}"
                        )
                    )

def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session() -> Session:
    """Get database session (for non-generator use)"""
    return SessionLocal()
