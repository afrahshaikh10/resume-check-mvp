from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Connect to SQLite database (creates file automatically)
engine = create_engine("sqlite:///resume_results.db")
Base = declarative_base()

# Table definition
class ResumeResult(Base):
    __tablename__ = "resumes"
    id = Column(Integer, primary_key=True)
    candidate_name = Column(String)
    resume_text = Column(Text)
    jd_text = Column(Text)
    relevance_score = Column(Float)
    hard_score = Column(Float)
    soft_score = Column(Float)
    verdict = Column(String)
    missing_skills = Column(Text)
    strengths = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# Create table
Base.metadata.create_all(engine)

# Optional: create a session object
Session = sessionmaker(bind=engine)
session = Session()

print("✅ Database and table created successfully!")
