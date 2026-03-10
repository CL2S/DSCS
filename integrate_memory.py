import json
import re
import datetime
from typing import List, Dict, Any
from memory_representation import (
    PatientRegistry, PatientRecord, PatientState,
    DoctorProfile, Prescription, FeedbackLog, FeedbackEvent,
    MedicalKnowledgeBase
)

def parse_values_from_text(text: str, key: str) -> List[float]:
    """
    Parses a list of numerical values from the text for a given key.
    Format example: "心率变化：[108.0, 108.0]（单位：次/分）"
    """
    # Escape parenthesis for regex
    pattern = f"{key}.*?\[(.*?)\]"
    match = re.search(pattern, text)
    if match:
        content = match.group(1)
        if not content.strip():
            return []
        try:
            # Handle "N/A" or other non-numeric values if necessary, 
            # but based on previous files, they seem to be numbers or strings of numbers.
            return [float(x.strip()) for x in content.split(',') if x.strip() and x.strip() != 'nan']
        except ValueError:
            return []
    return []

def extract_basic_info(text: str) -> Dict[str, Any]:
    """
    Extracts basic info like age, gender, weight from the description text.
    Example: "ICU住院编号 30117609，对应患者编号 14413751，43岁男性，体重71.4kg。"
    """
    info = {}
    
    # Age and Gender
    age_gender_match = re.search(r'(\d+)岁(男性|女性)', text)
    if age_gender_match:
        info['age'] = int(age_gender_match.group(1))
        info['gender'] = 'M' if age_gender_match.group(2) == '男性' else 'F'
    
    # Weight
    weight_match = re.search(r'体重([\d\.]+)kg', text)
    if weight_match:
        info['weight'] = float(weight_match.group(1))
        
    # Stay ID and Subject ID
    stay_id_match = re.search(r'ICU住院编号 (\d+)', text)
    if stay_id_match:
        info['stay_id'] = stay_id_match.group(1)
        
    subject_id_match = re.search(r'对应患者编号 (\d+)', text)
    if subject_id_match:
        info['subject_id'] = subject_id_match.group(1)
        
    return info

def integrate_dataset(json_file_path: str):
    print(f"Loading data from {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Initialize Memory Components
    patient_registry = PatientRegistry()
    
    # Create a mock doctor for the AI system
    ai_doctor = DoctorProfile(
        doctor_id="AI_CLINICIAN_001",
        name="AI System",
        position="System",
        specialty="Critical Care"
    )
    
    # Knowledge Base (Mock initialization)
    kb = MedicalKnowledgeBase()
    kb.register_entity("medication", "Norepinephrine")
    kb.register_entity("medication", "Dopamine")
    
    print(f"Processing {len(data)} patient records...")
    
    for entry in data:
        input_desc = entry.get('input_description', '')
        output_summary = entry.get('output_summary', '')
        stay_id = str(entry.get('stay_id'))
        
        # 1. Extract Patient Info & State
        basic_info = extract_basic_info(input_desc)
        basic_info['diagnosis'] = 'Sepsis' # Implicit context
        
        # Extract time-series data from input description (Pre-ICU / Initial)
        # For simplicity, we extract a few key metrics
        hr_values = parse_values_from_text(input_desc, "心率变化")
        map_values = parse_values_from_text(input_desc, "平均动脉压变化")
        sofa_values = parse_values_from_text(input_desc, "SOFA总分变化")
        
        # Construct time series data
        # Assuming all lists have same length and correspond to time steps
        # We'll just take the max length found
        max_len = max(len(hr_values), len(map_values), len(sofa_values)) if any([hr_values, map_values, sofa_values]) else 0
        
        time_series = []
        for t in range(max_len):
            ts_point = {
                'time_step': t,
                'heart_rate': hr_values[t] if t < len(hr_values) else None,
                'map': map_values[t] if t < len(map_values) else None,
                'sofa_total': sofa_values[t] if t < len(sofa_values) else None
            }
            time_series.append(ts_point)
            
        # Create Patient Record
        patient_state = PatientState(basic_info, time_series)
        patient_record = PatientRecord(patient_id=stay_id, state=patient_state)
        
        # Mock embedding (in reality this comes from the encoder)
        patient_record.update_embedding([0.1, 0.2, 0.3]) 
        
        patient_registry.add_patient(patient_record)
        
        # 2. Extract Intervention & Outcome from Output Summary (Post-ICU / ICU duration)
        # Extract Vasopressor usage as Intervention
        vaso_values = parse_values_from_text(output_summary, "血管活性药物使用剂量")
        sofa_post = parse_values_from_text(output_summary, "SOFA总分变化") # If available in output summary text? 
        # Note: The example output summary text provided in tool output didn't explicitly show "SOFA总分变化" text, 
        # but the JSON structure has 'sofa_scores_post_icu' if we look deeper or generated code.
        # Let's check "oxygenation index" etc. from text.
        
        # Let's assume intervention is "Vasopressor" if max dose > 0
        max_vaso = max(vaso_values) if vaso_values else 0.0
        
        intervention_type = "Vasopressor" if max_vaso > 0 else "Observation"
        
        # Create Prescription
        prescription = Prescription(
            patient_id=stay_id,
            intervention={
                'type': intervention_type,
                'max_dose': max_vaso,
                'details': f"Max vasopressor dose: {max_vaso}"
            }
        )
        ai_doctor.add_prescription(prescription)
        
        # 3. Create Feedback Event (Outcome)
        # Outcome: Did SOFA score improve?
        # We need SOFA scores from the output phase.
        # The JSON object has 'sofa_scores_post_icu' (implied from generator code) or we parse from text if available.
        # Let's try to find it in the text or fallback to 0.
        # The sample text showed '氧合指数' etc but not explicitly 'SOFA总分' in the text body printed.
        # However, we can use the 'sofa_scores' field from the JSON entry if it exists.
        
        outcome_desc = "Unknown"
        sofa_scores_list = entry.get('sofa_scores', [])
        # In the provided JSON sample, 'sofa_scores' seems to be the input/pre-ICU or maybe combined?
        # The generator code separates them. Let's look at the JSON structure again.
        # The sample shows "sofa_scores" as a list of dicts.
        
        if sofa_scores_list:
            initial_sofa = sofa_scores_list[0].get('sofa_total', 0)
            final_sofa = sofa_scores_list[-1].get('sofa_total', 0)
            
            if final_sofa < initial_sofa:
                outcome_desc = "Improved"
            elif final_sofa > initial_sofa:
                outcome_desc = "Deteriorated"
            else:
                outcome_desc = "Stable"
        
        feedback_event = FeedbackEvent(
            event_id=f"FB_{stay_id}",
            doctor_id=ai_doctor.doctor_id,
            patient_id=stay_id,
            intervention_type=intervention_type,
            timestamp=datetime.datetime.now(),
            outcome=outcome_desc,
            knowledge_impact="Routine update"
        )
        
        ai_doctor.record_feedback(feedback_event)

    print("Integration complete.")
    print(f"Total Patients: {len(patient_registry.patients)}")
    print(f"Total Prescriptions: {len(ai_doctor.prescriptions)}")
    print(f"Total Feedback Events: {len(ai_doctor.decision_trajectory.events)}")
    
    # Example verification
    first_pid = list(patient_registry.patients.keys())[0]
    print(f"\nSample Patient ({first_pid}):")
    print(patient_registry.get_patient(first_pid).state.basic_info)
    print(f"Latest Metrics: {patient_registry.get_patient(first_pid).state.get_latest_metrics()}")

if __name__ == "__main__":
    integrate_dataset("icu_stays_descriptions88.json")
