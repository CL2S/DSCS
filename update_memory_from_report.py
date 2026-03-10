import json
import re
import datetime
from typing import Dict, Any, List, Optional
from memory_evolution import MemoryEvolutionSystem, FeedbackEvaluator, ROME_Editor, MOOM_Updater, MemoryLLM_Manager
from memory_representation import MedicalKnowledgeBase, FeedbackEvent, DoctorProfile, PatientRecord, PatientState

# -------------------------------------------------------------------------
# Report Parsing & Feedback Event Extraction
# -------------------------------------------------------------------------

class ReportParser:
    """
    Parses evaluator report JSON to extract feedback events and context.
    """
    def __init__(self, report_path: str):
        self.report_path = report_path
        self.data = self._load_report()

    def _load_report(self) -> Dict[str, Any]:
        with open(self.report_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_context(self) -> Dict[str, Any]:
        """
        Extracts patient state and intervention context.
        """
        # Parse Input Description for Patient State
        input_desc = self.data.get('input_description', '')
        # Simplified extraction logic (similar to integrate_memory.py)
        # In a real system, this would reuse the same parser
        patient_id = self.data.get('patient_id')
        
        # Create a mock patient state for evaluation context
        patient_state = {
            'patient_id': patient_id,
            'description': input_desc,
            # We could parse vital signs here if needed by the evaluator
            'metrics': {} 
        }

        # Create a mock doctor profile (the system acting as doctor)
        doctor_profile = {
            'doctor_id': 'AI_EVALUATOR',
            'success_rate': 0.8, # Default or from history
            'specialty': 'Critical Care'
        }

        # Knowledge Base context (mock)
        knowledge_base = {
            'rules_count': 100,
            'entropy': 0.5
        }

        return {
            'patient_state': patient_state,
            'doctor_profile': doctor_profile,
            'knowledge_base': knowledge_base
        }

    def extract_feedback_event(self) -> FeedbackEvent:
        """
        Creates a FeedbackEvent based on the evaluation result.
        """
        # Determine outcome based on confidence and evaluation content
        confidence = self.data.get('confidence_score', 0.0)
        eval_text = self.data.get('evaluation', {}).get('model_output', '')
        
        # Simple heuristic for outcome classification
        outcome = "UNCERTAIN"
        if confidence > 0.8:
            outcome = "HIGH_CONFIDENCE_SUCCESS"
        elif confidence < 0.4:
            outcome = "LOW_CONFIDENCE_FAILURE"
        
        # Check for critical keywords in evaluation text
        urgency = "NORMAL"
        if "risk" in eval_text.lower() or "critical" in eval_text.lower() or "warning" in eval_text.lower():
            urgency = "CRITICAL"

        # Create event
        event = {
            'event_id': f"EVT_{self.data.get('timestamp')}",
            'doctor_id': 'AI_EVALUATOR',
            'patient_id': self.data.get('patient_id'),
            'intervention_type': 'Vasopressor', # Extracted from intervention text
            'timestamp': datetime.datetime.now(),
            'outcome': outcome,
            'content': eval_text, # The full evaluation reasoning
            'urgency': urgency,
            'concept': 'Sepsis_Management', # Target concept for ROME
            'p_value': 0.01 if confidence > 0.9 else 0.5 # Mock p-value derivation
        }
        
        return event

# -------------------------------------------------------------------------
# Integration Logic
# -------------------------------------------------------------------------

def update_knowledge_from_report(report_path: str):
    print(f"Processing report: {report_path}")
    
    # 1. Parse Report
    parser = ReportParser(report_path)
    context = parser.extract_context()
    feedback_event_dict = parser.extract_feedback_event()
    
    print(f"Extracted Event: {feedback_event_dict['event_id']}, Outcome: {feedback_event_dict['outcome']}, Urgency: {feedback_event_dict['urgency']}")

    # 2. Initialize Evolution System
    evolution_system = MemoryEvolutionSystem()
    
    # 3. Process Feedback
    print("Evaluating feedback for memory update...")
    result = evolution_system.process_feedback(feedback_event_dict, context)
    
    print(f"Update Result: {result}")
    
    if result['status'] == 'PENDING_VERIFICATION':
        print("Triggering Nightly Batch Process...")
        evolution_system.nightly_batch_process()
    
    elif result['status'] == 'URGENT_UPDATE':
        print("Urgent update applied immediately via ROME.")

if __name__ == "__main__":
    REPORT_PATH = "/data/wzx/output/deepseek-r1/evaluator_30315020_deepseek-r1_32b_20251225_112800.json"
    update_knowledge_from_report(REPORT_PATH)
