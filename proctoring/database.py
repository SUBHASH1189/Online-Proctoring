from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Default DB URL (for production)
SQLALCHEMY_DATABASE_URL = 'postgresql+psycopg2://postgres:subhash@localhost:5432/Proctoring'

# These will be overridden in test setup
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()