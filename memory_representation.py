import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any, Union

# -------------------------------------------------------------------------
# 1.1 患者信息 (Patient / PatientRecord)
# -------------------------------------------------------------------------

class PatientState:
    """
    保存患者基本信息与后续监测生理指标。
    """
    def __init__(self, basic_info: Dict[str, Any], time_series_data: List[Dict[str, Any]]):
        self.basic_info = basic_info  # e.g., {'age': 65, 'gender': 'M', 'diagnosis': 'Sepsis'}
        self.time_series_data = time_series_data  # e.g., [{'time': 0, 'MAP': 70, 'SOFA': 5}, ...]

    def get_latest_metrics(self) -> Dict[str, Any]:
        """获取最新的生理指标"""
        if not self.time_series_data:
            return {}
        return self.time_series_data[-1]

    def update_metrics(self, timestamp: int, metrics: Dict[str, Any]):
        """更新时间步的生理指标"""
        metrics['time'] = timestamp
        self.time_series_data.append(metrics)


class PatientRecord:
    """
    患者记录实体，包含多模态嵌入、概念激活度及状态数据。
    """
    def __init__(self, patient_id: str, state: PatientState):
        self.patient_id = patient_id
        self.state = state
        self.embedding: Optional[List[float]] = None  # 多模态时序编码器输出的患者嵌入向量
        self.concept_activations: Optional[Dict[str, float]] = None  # 临床概念激活度

    def update_embedding(self, embedding: List[float]):
        self.embedding = embedding

    def update_concepts(self, concepts: Dict[str, float]):
        self.concept_activations = concepts


class PatientRegistry:
    """
    患者集合管理（列表+索引）。
    """
    def __init__(self):
        self.patients: Dict[str, PatientRecord] = {}

    def add_patient(self, patient: PatientRecord):
        self.patients[patient.patient_id] = patient

    def get_patient(self, patient_id: str) -> Optional[PatientRecord]:
        return self.patients.get(patient_id)


# -------------------------------------------------------------------------
# 1.2 医生决策修正轨迹 (Doctor / DoctorProfile)
# -------------------------------------------------------------------------

class FeedbackEvent:
    """
    单条反馈事件，记录决策修正细节。
    """
    def __init__(self, 
                 event_id: str, 
                 doctor_id: str, 
                 patient_id: str, 
                 intervention_type: str, 
                 timestamp: datetime, 
                 outcome: str,
                 knowledge_impact: Optional[str] = None):
        self.event_id = event_id
        self.doctor_id = doctor_id
        self.patient_id = patient_id
        self.intervention_type = intervention_type
        self.timestamp = timestamp
        self.outcome = outcome
        self.knowledge_impact = knowledge_impact


class FeedbackLog:
    """
    决策历史日志。
    """
    def __init__(self):
        self.events: List[FeedbackEvent] = []

    def add_event(self, event: FeedbackEvent):
        self.events.append(event)

    def search_similar_feedback(self, intervention_type: str, diagnosis: str) -> List[FeedbackEvent]:
        """
        按“同类干预 + 同类诊断”检索相似反馈（简化版逻辑）。
        实际应用中需要结合 PatientRecord 查询诊断。
        """
        # 这里仅演示接口定义，实际需关联 PatientRegistry 查询诊断
        return [e for e in self.events if e.intervention_type == intervention_type]


class Prescription:
    """
    处方干预方案。
    """
    def __init__(self, patient_id: str, intervention: Dict[str, Any]):
        self.patient_id = patient_id
        # intervention 包含: {'medication': '...', 'operation': '...', 'time': '...'}
        self.intervention = intervention


class DoctorProfile:
    """
    医生实体，包含个人信息、处方记录及决策轨迹。
    """
    def __init__(self, doctor_id: str, name: str, position: str, specialty: str):
        self.doctor_id = doctor_id
        self.personal_info = {
            'name': name,
            'position': position,  # e.g., 'Chief Physician', 'Attending Physician'
            'specialty': specialty,
            'success_rate': 0.0  # 动态更新
        }
        self.prescriptions: List[Prescription] = []
        self.decision_trajectory = FeedbackLog()

    def add_prescription(self, prescription: Prescription):
        self.prescriptions.append(prescription)

    def record_feedback(self, event: FeedbackEvent):
        self.decision_trajectory.add_event(event)
        # TODO: 触发 success_rate 更新逻辑

    def update_success_rate(self, new_rate: float):
        self.personal_info['success_rate'] = new_rate


# -------------------------------------------------------------------------
# 1.3 知识存储 (MedicalKnowledgeBase)
# -------------------------------------------------------------------------

class MedicalKnowledgeBase:
    """
    医学知识库，存储规则、实体库及约束。
    """
    def __init__(self):
        self.clinical_rules: List[Dict[str, Any]] = []  # e.g., [{'condition': 'Si', 'action': 'Inv', 'next_state': 'Sj'}]
        self.symbolic_constraints: List[str] = []  # 符号约束规则
        
        # 实体库
        self.disease_library: List[str] = []
        self.medication_library: List[str] = []
        self.procedure_library: List[str] = []

    def add_rule(self, rule: Dict[str, Any]):
        """插入规则前的一致性验证"""
        if self.validate_rule(rule):
            self.clinical_rules.append(rule)
            print(f"Rule added: {rule}")
        else:
            print(f"Rule validation failed: {rule}")

    def validate_rule(self, rule: Dict[str, Any]) -> bool:
        """
        检查规则中的 intervention 是否在药物库内等约束。
        """
        intervention = rule.get('action')
        if intervention and intervention not in self.medication_library and intervention not in self.procedure_library:
            return False
        return True

    def register_entity(self, entity_type: str, name: str):
        """注册实体到对应的库"""
        if entity_type == 'disease':
            self.disease_library.append(name)
        elif entity_type == 'medication':
            self.medication_library.append(name)
        elif entity_type == 'procedure':
            self.procedure_library.append(name)
