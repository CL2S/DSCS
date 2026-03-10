#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Celery tasks for the sepsis prediction web application.

This module provides:
1. Celery application configuration
2. Async task definitions for prediction and evaluation
3. Task status tracking and progress updates
4. Integration with database for task persistence
"""

import os
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

# Try to import Celery, with fallback for development
try:
    from celery import Celery, Task
    from celery.exceptions import Ignore
    CELERY_AVAILABLE = True
except ImportError:
    print("Warning: Celery not available. Running in synchronous mode.")
    CELERY_AVAILABLE = False
    # Mock Celery classes for development
    class Celery:
        def __init__(self, *args, **kwargs):
            pass
    class Task:
        def __init__(self, *args, **kwargs):
            pass
        def update_state(self, *args, **kwargs):
            pass

# Import database modules
try:
    from database import (
        SessionLocal,
        Prediction, PredictionResult,
        PredictionCRUD, PredictionResultCRUD
    )
    DATABASE_AVAILABLE = True
except ImportError:
    print("Warning: Database module not available.")
    DATABASE_AVAILABLE = False

# Import core functions
try:
    from core_functions import (
        run_prediction,
        run_evaluation,
        select_best_prediction,
        save_best_prediction_result
    )
    CORE_FUNCTIONS_AVAILABLE = True
except ImportError:
    print("Warning: Core functions not available.")
    CORE_FUNCTIONS_AVAILABLE = False

# Import patient ID extraction
try:
    from sofa_prediction_evaluator import extract_patient_id
    EXTRACT_ID_AVAILABLE = True
except ImportError:
    print("Warning: Patient ID extraction not available.")
    EXTRACT_ID_AVAILABLE = False
    def extract_patient_id(*args, **kwargs):
        return "unknown"

# ====================== Celery Configuration ======================

# Redis URL for Celery broker and backend
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

if CELERY_AVAILABLE:
    # Create Celery application
    celery_app = Celery(
        "sepsis_tasks",
        broker=REDIS_URL,
        backend=REDIS_URL,
        include=["tasks"]
    )

    # Celery configuration
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_send_sent_event=True,
        worker_send_task_events=True,
        task_ignore_result=False,
        task_store_errors_even_if_ignored=True,
        # Task settings
        task_time_limit=300,  # 5 minutes
        task_soft_time_limit=270,  # 4.5 minutes
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=100,
        # Result expiration (1 hour)
        result_expires=3600,
    )
else:
    celery_app = None


# ====================== Task Base Class ======================

class DatabaseTask:
    """Mixin for tasks that need database access."""

    def update_task_progress(self, task_id: str, progress: float, status: str = "running", error: str = None):
        """Update task progress in database."""
        if not DATABASE_AVAILABLE:
            return False

        try:
            db = SessionLocal()
            prediction = PredictionCRUD.get_by_task_id(db, task_id)
            if prediction:
                PredictionCRUD.update_status(
                    db, prediction.id, status,
                    progress=progress, error=error
                )
            db.close()
            return True
        except Exception as e:
            print(f"Error updating task progress: {e}")
            return False

    def save_prediction_result(self, task_id: str, result_data: Dict[str, Any]) -> Optional[int]:
        """Save prediction result to database."""
        if not DATABASE_AVAILABLE:
            return None

        try:
            db = SessionLocal()

            # Get prediction by task ID
            prediction = PredictionCRUD.get_by_task_id(db, task_id)
            if not prediction:
                db.close()
                return None

            # Create prediction result
            result_record = {
                "prediction_model": result_data.get("prediction_model"),
                "evaluation_models": result_data.get("evaluation_models", []),
                "total_confidence": result_data.get("total_confidence", 0.0),
                "prediction_data": result_data.get("prediction_data", {}),
                "prediction_file": result_data.get("prediction_file"),
                "result_file": result_data.get("result_file", ""),
                "predicted_sofa_scores": result_data.get("predicted_sofa_scores"),
                "predicted_sofa_totals": result_data.get("predicted_sofa_totals"),
                "risk_assessment": result_data.get("risk_assessment"),
                "chart_data": result_data.get("chart_data")
            }

            result = PredictionResultCRUD.create(db, result_record)

            # Link prediction to result
            PredictionCRUD.update_result(db, prediction.id, result.id)

            db.close()
            return result.id
        except Exception as e:
            print(f"Error saving prediction result: {e}")
            if 'db' in locals():
                db.close()
            return None


# ====================== Task Definitions ======================

if CELERY_AVAILABLE:
    @celery_app.task(bind=True, base=Task)
    def predict_single_task(self, prediction_id: int, patient_data: Dict[str, Any], model_name: str):
        """Celery task for single model prediction."""
        task_id = self.request.id

        # Update task status to running
        if hasattr(self, 'update_task_progress'):
            self.update_task_progress(task_id, 0.1, "running")

        if not CORE_FUNCTIONS_AVAILABLE:
            error_msg = "Core functions not available"
            self.update_task_progress(task_id, 0, "failed", error_msg)
            raise Exception(error_msg)

        try:
            # Step 1: Run prediction (30% progress)
            self.update_task_progress(task_id, 0.3, "running", "Running prediction...")
            prediction_file = run_prediction(model_name, patient_data)

            if not prediction_file:
                error_msg = "Prediction failed"
                self.update_task_progress(task_id, 0, "failed", error_msg)
                raise Exception(error_msg)

            # Step 2: Run evaluation (60% progress)
            self.update_task_progress(task_id, 0.6, "running", "Evaluating prediction...")
            confidence = run_evaluation(model_name, prediction_file)

            # Step 3: Load prediction data (80% progress)
            self.update_task_progress(task_id, 0.8, "running", "Processing results...")
            with open(prediction_file, "r", encoding="utf-8") as f:
                prediction_data = json.load(f)

            # Step 4: Prepare result (90% progress)
            self.update_task_progress(task_id, 0.9, "running", "Saving results...")

            # Extract patient ID for result file
            patient_id = "unknown"
            if EXTRACT_ID_AVAILABLE:
                patient_id = extract_patient_id(
                    patient_data.get("input_description", ""),
                    patient_data
                )

            result_file = f"./output/best_result/result_{patient_id}.json"

            # Prepare result data
            best_result = {
                "prediction_model": model_name,
                "evaluation_models": [model_name],
                "total_confidence": confidence if confidence is not None else 0,
                "prediction_file": prediction_file,
                "prediction_data": prediction_data,
                "output_summary": patient_data.get("output_summary", {}),
                "result_file": result_file
            }

            # Save result to file
            save_best_prediction_result(best_result, result_file)

            # Save to database
            if hasattr(self, 'save_prediction_result'):
                self.save_prediction_result(task_id, best_result)

            # Step 5: Complete (100% progress)
            self.update_task_progress(task_id, 1.0, "completed")

            return {
                "status": "success",
                "prediction_file": prediction_file,
                "result_file": result_file,
                "confidence": confidence,
                "patient_id": patient_id
            }

        except Exception as e:
            error_msg = str(e)
            self.update_task_progress(task_id, 0, "failed", error_msg)
            raise

    @celery_app.task(bind=True, base=Task)
    def predict_auto_task(self, prediction_id: int, patient_data: Dict[str, Any], model_names: List[str]):
        """Celery task for auto mode prediction (multiple models)."""
        task_id = self.request.id

        # Update task status to running
        if hasattr(self, 'update_task_progress'):
            self.update_task_progress(task_id, 0.1, "running")

        if not CORE_FUNCTIONS_AVAILABLE:
            error_msg = "Core functions not available"
            self.update_task_progress(task_id, 0, "failed", error_msg)
            raise Exception(error_msg)

        try:
            # Step 1: Run auto prediction (30% progress)
            self.update_task_progress(task_id, 0.3, "running", "Running auto prediction with multiple models...")
            best_result = select_best_prediction(model_names, patient_data)

            if not best_result:
                # Fallback to first model
                self.update_task_progress(task_id, 0.4, "running", "Auto mode failed, trying single model...")
                fallback_model = model_names[0] if model_names else None
                if not fallback_model:
                    error_msg = "No models available"
                    self.update_task_progress(task_id, 0, "failed", error_msg)
                    raise Exception(error_msg)

                prediction_file = run_prediction(fallback_model, patient_data)
                if not prediction_file:
                    error_msg = "Fallback prediction failed"
                    self.update_task_progress(task_id, 0, "failed", error_msg)
                    raise Exception(error_msg)

                confidence = run_evaluation(fallback_model, prediction_file)
                with open(prediction_file, "r", encoding="utf-8") as f:
                    prediction_data = json.load(f)

                best_result = {
                    "prediction_model": fallback_model,
                    "evaluation_models": [fallback_model],
                    "total_confidence": confidence if confidence is not None else 0,
                    "prediction_file": prediction_file,
                    "prediction_data": prediction_data,
                    "output_summary": patient_data.get("output_summary", {})
                }

            # Step 2: Extract patient ID and save result (80% progress)
            self.update_task_progress(task_id, 0.8, "running", "Saving results...")

            patient_id = "unknown"
            if EXTRACT_ID_AVAILABLE:
                patient_id = extract_patient_id(
                    patient_data.get("input_description", ""),
                    patient_data
                )

            result_file = f"./output/best_result/result_{patient_id}.json"
            best_result["result_file"] = result_file

            # Save result to file
            save_best_prediction_result(best_result, result_file)

            # Save to database
            if hasattr(self, 'save_prediction_result'):
                self.save_prediction_result(task_id, best_result)

            # Step 3: Complete (100% progress)
            self.update_task_progress(task_id, 1.0, "completed")

            return {
                "status": "success",
                "result_file": result_file,
                "best_model": best_result.get("prediction_model"),
                "confidence": best_result.get("total_confidence"),
                "patient_id": patient_id
            }

        except Exception as e:
            error_msg = str(e)
            self.update_task_progress(task_id, 0, "failed", error_msg)
            raise

    @celery_app.task(bind=True, base=Task)
    def batch_predict_task(self, prediction_id: int, patient_ids: List[int], model_names: List[str], intervention: str):
        """Celery task for batch prediction of multiple patients."""
        task_id = self.request.id

        # Update task status to running
        if hasattr(self, 'update_task_progress'):
            self.update_task_progress(task_id, 0.05, "running")

        if not DATABASE_AVAILABLE:
            error_msg = "Database not available for batch processing"
            self.update_task_progress(task_id, 0, "failed", error_msg)
            raise Exception(error_msg)

        try:
            db = SessionLocal()
            results = []

            total_patients = len(patient_ids)
            for i, patient_id in enumerate(patient_ids):
                # Update progress
                progress = 0.05 + (i / total_patients) * 0.9
                self.update_task_progress(
                    task_id, progress, "running",
                    f"Processing patient {i+1}/{total_patients}"
                )

                # Get patient from database
                from database import PatientCRUD
                patient = PatientCRUD.get_by_id(db, patient_id)
                if not patient:
                    continue

                # Prepare patient data
                patient_data = {
                    "input_description": patient.input_description,
                    "intervention": intervention,
                    "vital_signs": patient.vital_signs or {},
                    "output_summary": patient.output_summary or {}
                }

                # Run prediction (simplified - could call predict_auto_task)
                try:
                    best_result = select_best_prediction(model_names, patient_data)
                    if best_result:
                        results.append({
                            "patient_id": patient_id,
                            "stay_id": patient.stay_id,
                            "best_model": best_result.get("prediction_model"),
                            "confidence": best_result.get("total_confidence"),
                            "success": True
                        })
                    else:
                        results.append({
                            "patient_id": patient_id,
                            "stay_id": patient.stay_id,
                            "success": False,
                            "error": "Prediction failed"
                        })
                except Exception as e:
                    results.append({
                        "patient_id": patient_id,
                        "stay_id": patient.stay_id,
                        "success": False,
                        "error": str(e)
                    })

                # Small delay to prevent overload
                time.sleep(0.5)

            db.close()

            # Complete (100% progress)
            self.update_task_progress(task_id, 1.0, "completed")

            return {
                "status": "success",
                "total_patients": total_patients,
                "processed": len([r for r in results if r.get("success", False)]),
                "failed": len([r for r in results if not r.get("success", True)]),
                "results": results
            }

        except Exception as e:
            error_msg = str(e)
            self.update_task_progress(task_id, 0, "failed", error_msg)
            raise

else:
    # Synchronous implementations when Celery is not available
    def predict_single_task(prediction_id: int, patient_data: Dict[str, Any], model_name: str):
        """Synchronous implementation of single prediction task."""
        print(f"Running synchronous prediction: prediction_id={prediction_id}, model={model_name}")

        if not CORE_FUNCTIONS_AVAILABLE:
            raise Exception("Core functions not available")

        # Run prediction synchronously
        prediction_file = run_prediction(model_name, patient_data)
        if not prediction_file:
            raise Exception("Prediction failed")

        confidence = run_evaluation(model_name, prediction_file)

        with open(prediction_file, "r", encoding="utf-8") as f:
            prediction_data = json.load(f)

        # Extract patient ID
        patient_id = "unknown"
        if EXTRACT_ID_AVAILABLE:
            patient_id = extract_patient_id(
                patient_data.get("input_description", ""),
                patient_data
            )

        result_file = f"./output/best_result/result_{patient_id}.json"

        # Prepare result data
        best_result = {
            "prediction_model": model_name,
            "evaluation_models": [model_name],
            "total_confidence": confidence if confidence is not None else 0,
            "prediction_file": prediction_file,
            "prediction_data": prediction_data,
            "output_summary": patient_data.get("output_summary", {}),
            "result_file": result_file
        }

        # Save result to file
        save_best_prediction_result(best_result, result_file)

        return {
            "status": "success",
            "prediction_file": prediction_file,
            "result_file": result_file,
            "confidence": confidence,
            "patient_id": patient_id
        }

    def predict_auto_task(prediction_id: int, patient_data: Dict[str, Any], model_names: List[str]):
        """Synchronous implementation of auto prediction task."""
        print(f"Running synchronous auto prediction: prediction_id={prediction_id}, models={model_names}")

        if not CORE_FUNCTIONS_AVAILABLE:
            raise Exception("Core functions not available")

        # Run auto prediction
        best_result = select_best_prediction(model_names, patient_data)

        if not best_result:
            # Fallback to first model
            fallback_model = model_names[0] if model_names else None
            if not fallback_model:
                raise Exception("No models available")

            prediction_file = run_prediction(fallback_model, patient_data)
            if not prediction_file:
                raise Exception("Fallback prediction failed")

            confidence = run_evaluation(fallback_model, prediction_file)
            with open(prediction_file, "r", encoding="utf-8") as f:
                prediction_data = json.load(f)

            best_result = {
                "prediction_model": fallback_model,
                "evaluation_models": [fallback_model],
                "total_confidence": confidence if confidence is not None else 0,
                "prediction_file": prediction_file,
                "prediction_data": prediction_data,
                "output_summary": patient_data.get("output_summary", {})
            }

        # Extract patient ID
        patient_id = "unknown"
        if EXTRACT_ID_AVAILABLE:
            patient_id = extract_patient_id(
                patient_data.get("input_description", ""),
                patient_data
            )

        result_file = f"./output/best_result/result_{patient_id}.json"
        best_result["result_file"] = result_file

        # Save result to file
        save_best_prediction_result(best_result, result_file)

        return {
            "status": "success",
            "result_file": result_file,
            "best_model": best_result.get("prediction_model"),
            "confidence": best_result.get("total_confidence"),
            "patient_id": patient_id
        }


# ====================== Task Management Utilities ======================

def create_prediction_task(patient_id: int, model_names: List[str], intervention: str,
                          prediction_mode: str = "single", parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a new prediction task and return task information."""

    if not DATABASE_AVAILABLE:
        return {
            "status": "error",
            "message": "Database not available",
            "task_id": None
        }

    try:
        db = SessionLocal()

        # Generate task ID
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        # Create prediction record
        prediction_data = {
            "task_id": task_id,
            "patient_id": patient_id,
            "model_names": model_names,
            "intervention": intervention,
            "prediction_mode": prediction_mode,
            "prediction_parameters": parameters or {},
            "status": "pending",
            "progress": 0.0
        }

        prediction = PredictionCRUD.create(db, prediction_data)
        db.close()

        return {
            "status": "success",
            "task_id": task_id,
            "prediction_id": prediction.id,
            "message": "Prediction task created successfully"
        }

    except Exception as e:
        print(f"Error creating prediction task: {e}")
        if 'db' in locals():
            db.close()
        return {
            "status": "error",
            "message": str(e),
            "task_id": None
        }


def start_prediction_task(prediction_id: int, patient_data: Dict[str, Any] = None):
    """Start a prediction task (runs synchronously or asynchronously)."""

    if not DATABASE_AVAILABLE:
        return {
            "status": "error",
            "message": "Database not available"
        }

    try:
        db = SessionLocal()

        # Get prediction
        prediction = PredictionCRUD.get_by_id(db, prediction_id)
        if not prediction:
            db.close()
            return {
                "status": "error",
                "message": "Prediction not found"
            }

        # If patient_data is not provided, get from database
        if patient_data is None:
            from database import PatientCRUD
            patient = PatientCRUD.get_by_id(db, prediction.patient_id)
            if not patient:
                db.close()
                return {
                    "status": "error",
                    "message": "Patient not found"
                }

            patient_data = {
                "input_description": patient.input_description,
                "intervention": prediction.intervention,
                "vital_signs": patient.vital_signs or {},
                "output_summary": patient.output_summary or {}
            }

        db.close()

        # Start the appropriate task based on mode
        if prediction.prediction_mode == "single":
            if len(prediction.model_names) == 0:
                return {
                    "status": "error",
                    "message": "No model specified for single mode"
                }
            model_name = prediction.model_names[0]

            if CELERY_AVAILABLE:
                # Start Celery task
                task = predict_single_task.delay(prediction_id, patient_data, model_name)
                return {
                    "status": "success",
                    "message": "Celery task started",
                    "celery_task_id": task.id,
                    "mode": "async"
                }
            else:
                # Run synchronously
                result = predict_single_task(prediction_id, patient_data, model_name)
                return {
                    "status": "success",
                    "message": "Task completed synchronously",
                    "result": result,
                    "mode": "sync"
                }

        elif prediction.prediction_mode == "auto":
            if len(prediction.model_names) < 1:
                return {
                    "status": "error",
                    "message": "At least one model required for auto mode"
                }

            if CELERY_AVAILABLE:
                # Start Celery task
                task = predict_auto_task.delay(prediction_id, patient_data, prediction.model_names)
                return {
                    "status": "success",
                    "message": "Celery task started",
                    "celery_task_id": task.id,
                    "mode": "async"
                }
            else:
                # Run synchronously
                result = predict_auto_task(prediction_id, patient_data, prediction.model_names)
                return {
                    "status": "success",
                    "message": "Task completed synchronously",
                    "result": result,
                    "mode": "sync"
                }

        else:
            return {
                "status": "error",
                "message": f"Unknown prediction mode: {prediction.prediction_mode}"
            }

    except Exception as e:
        print(f"Error starting prediction task: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get status of a task by ID."""

    if not DATABASE_AVAILABLE:
        return {
            "status": "unknown",
            "message": "Database not available"
        }

    try:
        db = SessionLocal()
        prediction = PredictionCRUD.get_by_task_id(db, task_id)

        if not prediction:
            db.close()
            return {
                "status": "not_found",
                "message": "Task not found"
            }

        result = {
            "task_id": prediction.task_id,
            "prediction_id": prediction.id,
            "status": prediction.status,
            "progress": prediction.progress,
            "error_message": prediction.error_message,
            "created_at": prediction.created_at.isoformat() if prediction.created_at else None,
            "started_at": prediction.started_at.isoformat() if prediction.started_at else None,
            "completed_at": prediction.completed_at.isoformat() if prediction.completed_at else None
        }

        # If task has a result, include result info
        if prediction.result_id:
            result_obj = PredictionResultCRUD.get_by_id(db, prediction.result_id)
            if result_obj:
                result["result"] = {
                    "confidence": result_obj.total_confidence,
                    "prediction_model": result_obj.prediction_model,
                    "created_at": result_obj.created_at.isoformat() if result_obj.created_at else None
                }

        db.close()
        return result

    except Exception as e:
        print(f"Error getting task status: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# ====================== Main Entry Point ======================

if __name__ == "__main__":
    """Test the task module."""
    print("Task module test")

    # Test database connection
    if DATABASE_AVAILABLE:
        print("✓ Database module available")
    else:
        print("✗ Database module not available")

    # Test Celery availability
    if CELERY_AVAILABLE:
        print("✓ Celery available")
    else:
        print("✗ Celery not available (running in synchronous mode)")

    # Test core functions
    if CORE_FUNCTIONS_AVAILABLE:
        print("✓ Core functions available")
    else:
        print("✗ Core functions not available")

    print("\nTask module initialized successfully.")