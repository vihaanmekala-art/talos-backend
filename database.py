from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os

# changed: normalize the configured URL once so every module shares the same database target
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/dbname")
if SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql://", 1)

# changed: tune the shared engine for lower checkout latency and healthier pooled connections
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_recycle=1800,
    pool_pre_ping=True,
    pool_use_lifo=True,
)
# changed: keep ORM instances from expiring on commit to avoid unnecessary refresh queries
SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)

Base = declarative_base()
