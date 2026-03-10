#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database module for the sepsis prediction web application.

This module provides:
1. SQLAlchemy ORM models for all database tables
2. Database connection and session management
3. CRUD operations for all entities
4. Database migration utilities
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import StaticPool

# Base class for declarative models
Base = declarative_base()

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./sepsis_prediction.db")

# Create engine with SQLite compatibility for development
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False  # Set to True for SQL query logging
    )
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Database dependency for FastAPI (if we migrate to FastAPI later)
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ====================== Database Models ======================

class User(Base):
    """User model for authentication and authorization."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(String(50), default="user")  # user, doctor, researcher, admin
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    predictions = relationship("Prediction", back_populates="user")


class Patient(Base):
    """Patient model for storing patient information."""
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    stay_id = Column(String(50), unique=True, index=True, nullable=False)
    subject_id = Column(String(50), index=True)
    age = Column(Integer)
    gender = Column(String(20))
    weight = Column(Float)

    # Clinical description and vital signs
    input_description = Column(Text, nullable=False)
    output_summary = Column(Text)
    vital_signs = Column(JSON)  # Store as JSON for flexibility

    # SOFA scores (stored as JSON for time series)
    sofa_scores = Column(JSON)
    sofa_scores_post_icu = Column(JSON)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    source_file = Column(String(255))
    notes = Column(Text)

    # Relationships
    predictions = relationship("Prediction", back_populates="patient")


class Prediction(Base):
    """Prediction task model."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(100), unique=True, index=True)  # Celery task ID

    # Foreign keys
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)

    # Input parameters
    model_names = Column(JSON)  # List of model names used
    intervention = Column(Text, nullable=False)
    prediction_mode = Column(String(50), default="single")  # single, auto, confidence, batch
    prediction_parameters = Column(JSON)  # Additional parameters

    # Status tracking
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    error_message = Column(Text)

    # Results reference (stored in prediction_results table)
    result_id = Column(Integer, ForeignKey("prediction_results.id"), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Relationships
    user = relationship("User", back_populates="predictions")
    patient = relationship("Patient", back_populates="predictions")
    result = relationship("PredictionResult", back_populates="prediction", uselist=False)


class PredictionResult(Base):
    """Detailed prediction results."""
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, index=True)

    # Core result data
    prediction_model = Column(String(100))
    evaluation_models = Column(JSON)  # List of evaluation models
    total_confidence = Column(Float)

    # Detailed prediction data (from core_functions output)
    prediction_data = Column(JSON)

    # SOFA predictions
    predicted_sofa_scores = Column(JSON)
    predicted_sofa_totals = Column(JSON)

    # Risk assessment
    risk_assessment = Column(JSON)

    # Visualization data
    chart_data = Column(JSON)  # Pre-processed data for frontend charts

    # File references
    prediction_file = Column(String(255))
    result_file = Column(String(255))

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    prediction = relationship("Prediction", back_populates="result", uselist=False)


class SystemConfig(Base):
    """System configuration and model settings."""
    __tablename__ = "system_config"

    id = Column(Integer, primary_key=True, index=True)
    config_key = Column(String(100), unique=True, index=True, nullable=False)
    config_value = Column(JSON, nullable=False)
    config_type = Column(String(50), default="string")  # string, number, boolean, json, list
    description = Column(Text)
    category = Column(String(50), default="general")
    is_editable = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ====================== CRUD Operations ======================

class PatientCRUD:
    """CRUD operations for Patient model."""

    @staticmethod
    def create(db: Session, patient_data: Dict[str, Any]) -> Patient:
        """Create a new patient record."""
        patient = Patient(**patient_data)
        db.add(patient)
        db.commit()
        db.refresh(patient)
        return patient

    @staticmethod
    def get_by_id(db: Session, patient_id: int) -> Optional[Patient]:
        """Get patient by ID."""
        return db.query(Patient).filter(Patient.id == patient_id).first()

    @staticmethod
    def get_by_stay_id(db: Session, stay_id: str) -> Optional[Patient]:
        """Get patient by stay ID."""
        return db.query(Patient).filter(Patient.stay_id == stay_id).first()

    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 100) -> List[Patient]:
        """Get all patients with pagination."""
        return db.query(Patient).offset(skip).limit(limit).all()

    @staticmethod
    def search(db: Session, query: str, skip: int = 0, limit: int = 100) -> List[Patient]:
        """Search patients by input_description or stay_id."""
        search_pattern = f"%{query}%"
        return db.query(Patient).filter(
            (Patient.input_description.ilike(search_pattern)) |
            (Patient.stay_id.ilike(search_pattern))
        ).offset(skip).limit(limit).all()

    @staticmethod
    def update(db: Session, patient_id: int, update_data: Dict[str, Any]) -> Optional[Patient]:
        """Update patient record."""
        patient = PatientCRUD.get_by_id(db, patient_id)
        if patient:
            for key, value in update_data.items():
                setattr(patient, key, value)
            patient.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(patient)
        return patient

    @staticmethod
    def delete(db: Session, patient_id: int) -> bool:
        """Delete patient record."""
        patient = PatientCRUD.get_by_id(db, patient_id)
        if patient:
            db.delete(patient)
            db.commit()
            return True
        return False


class PredictionCRUD:
    """CRUD operations for Prediction model."""

    @staticmethod
    def create(db: Session, prediction_data: Dict[str, Any]) -> Prediction:
        """Create a new prediction task."""
        prediction = Prediction(**prediction_data)
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        return prediction

    @staticmethod
    def get_by_id(db: Session, prediction_id: int) -> Optional[Prediction]:
        """Get prediction by ID."""
        return db.query(Prediction).filter(Prediction.id == prediction_id).first()

    @staticmethod
    def get_by_task_id(db: Session, task_id: str) -> Optional[Prediction]:
        """Get prediction by Celery task ID."""
        return db.query(Prediction).filter(Prediction.task_id == task_id).first()

    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 100) -> List[Prediction]:
        """Get all predictions with pagination."""
        return db.query(Prediction).offset(skip).limit(limit).all()

    @staticmethod
    def get_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[Prediction]:
        """Get predictions by user ID."""
        return db.query(Prediction).filter(Prediction.user_id == user_id).offset(skip).limit(limit).all()

    @staticmethod
    def get_by_patient(db: Session, patient_id: int, skip: int = 0, limit: int = 100) -> List[Prediction]:
        """Get predictions by patient ID."""
        return db.query(Prediction).filter(Prediction.patient_id == patient_id).offset(skip).limit(limit).all()

    @staticmethod
    def update_status(db: Session, prediction_id: int, status: str, progress: float = None, error: str = None) -> Optional[Prediction]:
        """Update prediction status."""
        prediction = PredictionCRUD.get_by_id(db, prediction_id)
        if prediction:
            prediction.status = status
            if progress is not None:
                prediction.progress = progress
            if error is not None:
                prediction.error_message = error

            if status == "running" and not prediction.started_at:
                prediction.started_at = datetime.utcnow()
            elif status in ["completed", "failed"] and not prediction.completed_at:
                prediction.completed_at = datetime.utcnow()

            db.commit()
            db.refresh(prediction)
        return prediction

    @staticmethod
    def update_result(db: Session, prediction_id: int, result_id: int) -> Optional[Prediction]:
        """Link prediction to its result."""
        prediction = PredictionCRUD.get_by_id(db, prediction_id)
        if prediction:
            prediction.result_id = result_id
            db.commit()
            db.refresh(prediction)
        return prediction


class PredictionResultCRUD:
    """CRUD operations for PredictionResult model."""

    @staticmethod
    def create(db: Session, result_data: Dict[str, Any]) -> PredictionResult:
        """Create a new prediction result."""
        result = PredictionResult(**result_data)
        db.add(result)
        db.commit()
        db.refresh(result)
        return result

    @staticmethod
    def get_by_id(db: Session, result_id: int) -> Optional[PredictionResult]:
        """Get result by ID."""
        return db.query(PredictionResult).filter(PredictionResult.id == result_id).first()

    @staticmethod
    def get_by_prediction(db: Session, prediction_id: int) -> Optional[PredictionResult]:
        """Get result by prediction ID."""
        prediction = PredictionCRUD.get_by_id(db, prediction_id)
        if prediction and prediction.result_id:
            return PredictionResultCRUD.get_by_id(db, prediction.result_id)
        return None


class SystemConfigCRUD:
    """CRUD operations for SystemConfig model."""

    @staticmethod
    def get(db: Session, config_key: str) -> Optional[SystemConfig]:
        """Get configuration by key."""
        return db.query(SystemConfig).filter(SystemConfig.config_key == config_key).first()

    @staticmethod
    def get_value(db: Session, config_key: str, default=None):
        """Get configuration value by key."""
        config = SystemConfigCRUD.get(db, config_key)
        if config:
            return config.config_value
        return default

    @staticmethod
    def set(db: Session, config_key: str, config_value, config_type: str = "json", description: str = None, category: str = "general") -> SystemConfig:
        """Set configuration value."""
        config = SystemConfigCRUD.get(db, config_key)
        if config:
            config.config_value = config_value
            config.config_type = config_type
            if description:
                config.description = description
            config.updated_at = datetime.utcnow()
        else:
            config = SystemConfig(
                config_key=config_key,
                config_value=config_value,
                config_type=config_type,
                description=description,
                category=category
            )
            db.add(config)
        db.commit()
        db.refresh(config)
        return config


# ====================== Database Initialization ======================

def init_db():
    """Initialize database: create all tables."""
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized: {DATABASE_URL}")

    # Create default system configurations
    db = SessionLocal()
    try:
        # Default model configurations
        SystemConfigCRUD.set(db, "available_models",
            ["gemma3:12b", "mistral:7b", "qwen3:4b", "qwen3:30b", "deepseek-r1:32b", "medllama2:latest"],
            config_type="list", description="Available Ollama models for prediction")

        SystemConfigCRUD.set(db, "default_models",
            ["gemma3:12b", "mistral:7b", "qwen3:4b"],
            config_type="list", description="Default models for auto mode")

        SystemConfigCRUD.set(db, "model_trust_scores_file",
            "/data/wzx/output/Model_Trust_Score.json",
            config_type="string", description="Path to model trust scores file")

        SystemConfigCRUD.set(db, "expert_interventions_file",
            "/data/wzx/extracted_expert_interventions.json",
            config_type="string", description="Path to expert interventions file")

        SystemConfigCRUD.set(db, "prediction_timeout",
            300,  # 5 minutes
            config_type="number", description="Prediction task timeout in seconds")

        SystemConfigCRUD.set(db, "max_concurrent_predictions",
            3,  # Maximum concurrent prediction tasks
            config_type="number", description="Maximum concurrent prediction tasks")

        print("Default system configurations created.")

    except Exception as e:
        print(f"Error creating default configurations: {e}")
        db.rollback()
    finally:
        db.close()


def drop_db():
    """Drop all database tables (for development only)."""
    Base.metadata.drop_all(bind=engine)
    print("Database dropped.")


def reset_db():
    """Reset database: drop and recreate all tables."""
    drop_db()
    init_db()


# ====================== Data Import Utilities ======================

def import_patients_from_json(db: Session, json_file: str = "./icu_stays_descriptions88.json"):
    """Import patients from JSON file."""
    import json
    from tqdm import tqdm

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            patients_data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return 0

    imported_count = 0
    for patient_data in tqdm(patients_data, desc="Importing patients"):
        try:
            # Extract basic information
            stay_id = str(patient_data.get("stay_id", ""))
            subject_id = str(patient_data.get("subject_id", ""))

            # Check if patient already exists
            existing = PatientCRUD.get_by_stay_id(db, stay_id)
            if existing:
                continue

            # Create patient record
            patient_record = {
                "stay_id": stay_id,
                "subject_id": subject_id,
                "input_description": patient_data.get("input_description", ""),
                "output_summary": patient_data.get("output_summary", ""),
                "sofa_scores": patient_data.get("sofa_scores"),
                "sofa_scores_post_icu": patient_data.get("sofa_scores_post_icu"),
                "source_file": json_file
            }

            # Try to extract age and gender from description
            description = patient_data.get("input_description", "")
            if "岁" in description:
                # Extract age (Chinese character for age)
                import re
                age_match = re.search(r'(\d+)岁', description)
                if age_match:
                    patient_record["age"] = int(age_match.group(1))

            if "男性" in description:
                patient_record["gender"] = "male"
            elif "女性" in description:
                patient_record["gender"] = "female"

            PatientCRUD.create(db, patient_record)
            imported_count += 1

        except Exception as e:
            print(f"Error importing patient {patient_data.get('stay_id')}: {e}")
            continue

    db.commit()
    print(f"Imported {imported_count} patients from {json_file}")
    return imported_count


# ====================== Main Entry Point ======================

if __name__ == "__main__":
    # Initialize database when run directly
    init_db()

    # Import patients from default JSON file
    db = SessionLocal()
    try:
        imported = import_patients_from_json(db)
        print(f"Total patients in database: {db.query(Patient).count()}")
    finally:
        db.close()