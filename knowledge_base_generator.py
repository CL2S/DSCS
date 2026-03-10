import pandas as pd
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
import os

class KnowledgeBaseBuilder:
    """
    Knowledge Base Builder based on the Technical Route Final Draft.
    Responsible for transforming raw ICU data into structured clinical cases.
    """
    
    def __init__(self, csv_path, output_path):
        self.csv_path = csv_path
        self.output_path = output_path
        self.dataset = None
        self.knowledge_base = []

    def load_data(self):
        """Load raw dataset from CSV."""
        print(f"Loading dataset from {self.csv_path}...")
        self.dataset = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.dataset)} records.")

    def _pad_feature_values(self, feature_dict):
        """Pad feature values to ensure consistent length for time-series data."""
        valid_lengths = [len(lst) for lst in feature_dict.values() 
                        if isinstance(lst, list) and len(lst) > 0]
        
        if not valid_lengths:
            return feature_dict
        
        max_len = max(valid_lengths)
        
        for key, values in feature_dict.items():
            if not isinstance(values, list):
                continue
                
            current_len = len(values)
            
            if current_len == 0:
                feature_dict[key] = ["N/A"] * max_len
                continue
                
            if current_len < max_len:
                pad_count = max_len - current_len
                last_value = values[-1]
                feature_dict[key] = values + [last_value] * pad_count
        
        return feature_dict

    def _format_list(self, data, unit="", prefix=""):
        """Format list of values into a readable string."""
        if not data:
            return ""
        return f"{prefix}[{', '.join(data)}]（单位：{unit}）"

    def _calculate_sofa_scores(self, row):
        """Calculate SOFA scores for a single time step."""
        # Respiration
        pao2_fio2 = row.get('pao2_fio2_ratio')
        ventilation = row.get('mechanical_ventilation', 0)
        if ventilation == 1: resp_score = 4
        elif pd.isna(pao2_fio2): resp_score = 0
        elif pao2_fio2 < 100: resp_score = 4
        elif pao2_fio2 < 200: resp_score = 3
        elif pao2_fio2 < 300: resp_score = 2
        elif pao2_fio2 < 400: resp_score = 1
        else: resp_score = 0

        # Coagulation
        platelet = row.get('platelet')
        if pd.isna(platelet): coag_score = 0
        elif platelet < 20: coag_score = 4
        elif platelet < 50: coag_score = 3
        elif platelet < 100: coag_score = 2
        elif platelet < 150: coag_score = 1
        else: coag_score = 0

        # Liver
        bilirubin = row.get('bilirubin_total')
        if pd.isna(bilirubin): liver_score = 0
        elif bilirubin >= 12.0: liver_score = 4
        elif bilirubin >= 6.0: liver_score = 3
        elif bilirubin >= 2.0: liver_score = 2
        elif bilirubin >= 1.2: liver_score = 1
        else: liver_score = 0

        # Cardiovascular
        map_val = row.get('map')
        vaso_rate = row.get('vasopressor_rate')
        if not pd.isna(vaso_rate) and vaso_rate > 0: cardio_score = 4
        elif pd.isna(map_val): cardio_score = 0
        elif map_val < 70: cardio_score = 1
        else: cardio_score = 0

        # CNS
        gcs = row.get('gcs_total')
        if pd.isna(gcs): cns_score = 0
        elif gcs <= 3: cns_score = 4
        elif gcs <= 6: cns_score = 3
        elif gcs <= 9: cns_score = 2
        elif gcs <= 12: cns_score = 1
        else: cns_score = 0

        # Renal
        creatinine = row.get('creatinine')
        urine = row.get('urine_output_ml')
        if not pd.isna(urine) and urine < 200: renal_score = 4
        elif not pd.isna(creatinine):
            if creatinine >= 5.0: renal_score = 4
            elif creatinine >= 3.5: renal_score = 3
            elif creatinine >= 2.0: renal_score = 2
            elif creatinine >= 1.2: renal_score = 1
            else: renal_score = 0
        else: renal_score = 0

        total_sofa = sum([resp_score, coag_score, liver_score, cardio_score, cns_score, renal_score])
        
        return {
            'sofa_respiration': resp_score,
            'sofa_coagulation': coag_score,
            'sofa_liver': liver_score,
            'sofa_cardiovascular': cardio_score,
            'sofa_cns': cns_score,
            'sofa_renal': renal_score,
            'sofa_total': total_sofa
        }

    def _fill_missing_values(self, data):
        """
        Fill missing values using data from adjacent time steps (-1 and 0).
        Logic aligned with Technical Route: ensure completeness of critical features.
        """
        sofa_related_features = [
            'pao2_fio2_ratio', 'mechanical_ventilation', 
            'platelet', 'bilirubin_total', 
            'map', 'vasopressor_rate', 
            'gcs_total', 'creatinine', 'urine_output_ml'
        ]
        
        minus1_data = data[data['time_step'] == -1].copy()
        zero_data = data[data['time_step'] == 0].copy()
        
        # Treat 0 urine output as missing (NaN)
        if 'urine_output_ml' in minus1_data.columns:
            minus1_data.loc[minus1_data['urine_output_ml'] == 0, 'urine_output_ml'] = float('nan')
        if 'urine_output_ml' in zero_data.columns:
            zero_data.loc[zero_data['urine_output_ml'] == 0, 'urine_output_ml'] = float('nan')
            
        has_minus1 = not minus1_data.empty
        has_zero = not zero_data.empty
        
        if not has_minus1 and not has_zero:
            return None
            
        # Check validity
        minus1_valid = False
        if has_minus1:
            avail_feats = [c for c in sofa_related_features if c in minus1_data.columns]
            minus1_valid = not minus1_data[avail_feats].isna().any().any()
            
        zero_valid = False
        if has_zero:
            avail_feats = [c for c in sofa_related_features if c in zero_data.columns]
            zero_valid = not zero_data[avail_feats].isna().any().any()
            
        if not minus1_valid and not zero_valid:
            return None
            
        data_copy = data.copy()
        
        # Fill logic
        if minus1_valid and not has_zero:
            zero_row = data[data['time_step'] == -1].iloc[0].copy()
            zero_row['time_step'] = 0
            data_copy = pd.concat([data_copy, pd.DataFrame([zero_row])], ignore_index=True)
            
        if zero_valid and not has_minus1:
            minus1_row = data[data['time_step'] == 0].iloc[0].copy()
            minus1_row['time_step'] = -1
            data_copy = pd.concat([data_copy, pd.DataFrame([minus1_row])], ignore_index=True)
            
        return data_copy

    def _generate_stay_description(self, stay_data):
        """Generate structured input description for a patient stay."""
        latest = stay_data.iloc[0]
        
        # Extract features
        sofa_features = {
            'pao2_fio2_ratio': stay_data['pao2_fio2_ratio'].dropna().round(2).astype(str).tolist(),
            'mechanical_ventilation': stay_data['mechanical_ventilation'].dropna().round(2).astype(str).tolist(),
            'platelet': stay_data['platelet'].dropna().round(2).astype(str).tolist(),
            'bilirubin_total': stay_data['bilirubin_total'].dropna().round(2).astype(str).tolist(),
            'map': stay_data['map'].dropna().round(2).astype(str).tolist(),
            'vasopressor_rate': stay_data['vasopressor_rate'].dropna().round(2).astype(str).tolist(),
            'gcs_total': stay_data.get('gcs_total', pd.Series()).dropna().round(2).astype(str).tolist(),
            'creatinine': stay_data['creatinine'].dropna().round(2).astype(str).tolist(),
            'urine_output_ml': stay_data['urine_output_ml'].dropna().round(2).astype(str).tolist()
        }
        
        vital_signs = {
            'heart_rate': stay_data['heart_rate'].dropna().round(2).astype(str).tolist(),
            'sbp': stay_data['sbp'].dropna().round(2).astype(str).tolist(),
            'dbp': stay_data['dbp'].dropna().round(2).astype(str).tolist(),
            'temperature': stay_data['temperature'].dropna().round(2).astype(str).tolist(),
            'resp_rate': stay_data['resp_rate'].dropna().round(2).astype(str).tolist(),
            'spo2': stay_data['spo2'].dropna().round(2).astype(str).tolist(),
            'map': stay_data['map'].dropna().round(2).astype(str).tolist(),
        }
        
        padded_sofa = self._pad_feature_values(sofa_features)
        padded_vitals = self._pad_feature_values(vital_signs)
        
        # Metadata
        meta = {
            'stay_id': str(latest['stay_id']),
            'subject_id': str(latest['subject_id']),
            'age': str(latest['age']),
            'gender': "男性" if latest['gender_male'] == 1 else "女性",
            'weight': str(latest.get('weight', 'N/A')),
            'duration': max(stay_data['time_step']) * 4,
            'measurements': len(stay_data)
        }
        
        # Construct Description Text
        desc = f"""ICU住院编号 {meta['stay_id']}，对应患者编号 {meta['subject_id']}，{meta['age']}岁{meta['gender']}，体重{meta['weight']}kg。

监测持续时间为 {meta['duration']} 小时（共 {meta['measurements']} 次测量，每4小时一次）：

【生命体征】
{self._format_list(padded_vitals['heart_rate'], "次/分", "心率变化：")}
{self._format_list(padded_vitals['sbp'], "mmHg", "收缩压变化：")}
{self._format_list(padded_vitals['dbp'], "mmHg", "舒张压变化：")}
{self._format_list(padded_vitals['temperature'], "℃", "体温变化：")}
{self._format_list(padded_vitals['resp_rate'], "次/分", "呼吸频率变化：")}
{self._format_list(padded_vitals['spo2'], "%", "血氧饱和度变化：")}
{self._format_list(padded_vitals['map'], "mmHg", "平均动脉压变化：")}

【SOFA评分相关特征】
{self._format_list(padded_sofa['pao2_fio2_ratio'], "mmHg", "氧合指数变化：")}
{self._format_list(padded_sofa['mechanical_ventilation'], "", "机械通气使用情况：")}
{self._format_list(padded_sofa['platelet'], "×10³/μL", "血小板计数变化：")}
{self._format_list(padded_sofa['bilirubin_total'], "mg/dL", "总胆红素变化：")}
{self._format_list(padded_sofa['vasopressor_rate'], "μg/kg/min", "血管活性药物使用剂量：")}
{self._format_list(padded_sofa['gcs_total'], "分", "格拉斯哥昏迷评分变化：")}
{self._format_list(padded_sofa['creatinine'], "mg/dL", "血清肌酐变化：")}
{self._format_list(padded_sofa['urine_output_ml'], "mL", "尿量变化：")}

【疾病评分】
{self._format_list(stay_data['sofa_total'].dropna().round(2).astype(str).tolist(), "分", "SOFA总分变化：")}
{self._format_list(stay_data.get('sirs_score', pd.Series()).dropna().round(2).astype(str).tolist(), "分", "SIRS评分变化：")}
""".strip()

        # Calculate Detailed SOFA Scores
        sofa_details = [self._calculate_sofa_scores(row) for _, row in stay_data.iterrows()]
        
        return {
            'description': desc,
            'sofa_details': sofa_details,
            'metadata': meta
        }

    def _generate_output_summary(self, stay_data):
        """Generate structured output summary (Post-ICU/Outcome)."""
        # Similar to input description but for post-ICU data
        sofa_features = {
            'pao2_fio2_ratio': stay_data['pao2_fio2_ratio'].dropna().round(2).astype(str).tolist(),
            'mechanical_ventilation': stay_data['mechanical_ventilation'].dropna().round(2).astype(str).tolist(),
            'platelet': stay_data['platelet'].dropna().round(2).astype(str).tolist(),
            'bilirubin_total': stay_data['bilirubin_total'].dropna().round(2).astype(str).tolist(),
            'map': stay_data['map'].dropna().round(2).astype(str).tolist(),
            'vasopressor_rate': stay_data['vasopressor_rate'].dropna().round(2).astype(str).tolist(),
            'gcs_total': stay_data.get('gcs_total', pd.Series()).dropna().round(2).astype(str).tolist(),
            'creatinine': stay_data['creatinine'].dropna().round(2).astype(str).tolist(),
            'urine_output_ml': stay_data['urine_output_ml'].dropna().round(2).astype(str).tolist()
        }
        
        vital_signs = {
            'heart_rate': stay_data['heart_rate'].dropna().round(2).astype(str).tolist(),
            'sbp': stay_data['sbp'].dropna().round(2).astype(str).tolist(),
            'dbp': stay_data['dbp'].dropna().round(2).astype(str).tolist(),
            'temperature': stay_data['temperature'].dropna().round(2).astype(str).tolist(),
            'resp_rate': stay_data['resp_rate'].dropna().round(2).astype(str).tolist(),
            'spo2': stay_data['spo2'].dropna().round(2).astype(str).tolist(),
            'map': stay_data['map'].dropna().round(2).astype(str).tolist(),
        }
        
        padded_sofa = self._pad_feature_values(sofa_features)
        padded_vitals = self._pad_feature_values(vital_signs)
        
        desc = f"""【ICU期间生命体征变化】
{self._format_list(padded_vitals['heart_rate'], "次/分", "心率变化：")}
{self._format_list(padded_vitals['sbp'], "mmHg", "收缩压变化：")}
{self._format_list(padded_vitals['dbp'], "mmHg", "舒张压变化：")}
{self._format_list(padded_vitals['temperature'], "℃", "体温变化：")}
{self._format_list(padded_vitals['resp_rate'], "次/分", "呼吸频率变化：")}
{self._format_list(padded_vitals['spo2'], "%", "血氧饱和度变化：")}
{self._format_list(padded_vitals['map'], "mmHg", "平均动脉压变化：")}

【ICU期间SOFA评分相关特征变化】
{self._format_list(padded_sofa['pao2_fio2_ratio'], "mmHg", "氧合指数变化：")}
{self._format_list(padded_sofa['mechanical_ventilation'], "", "机械通气使用情况：")}
{self._format_list(padded_sofa['platelet'], "×10³/μL", "血小板计数变化：")}
{self._format_list(padded_sofa['bilirubin_total'], "mg/dL", "总胆红素变化：")}
{self._format_list(padded_sofa['vasopressor_rate'], "μg/kg/min", "血管活性药物使用剂量：")}
{self._format_list(padded_sofa['gcs_total'], "分", "格拉斯哥昏迷评分变化：")}
{self._format_list(padded_sofa['creatinine'], "mg/dL", "血清肌酐变化：")}
{self._format_list(padded_sofa['urine_output_ml'], "mL", "尿量变化：")}
""".strip()

        sofa_details = [self._calculate_sofa_scores(row) for _, row in stay_data.iterrows()]
        
        return {
            'description': desc,
            'sofa_details': sofa_details
        }

    def build(self):
        """Execute the knowledge base construction process."""
        if self.dataset is None:
            self.load_data()
            
        stay_groups = self.dataset.groupby('stay_id')
        print(f"Processing {len(stay_groups)} unique ICU stays...")
        
        for stay_id, stay_data in stay_groups:
            stay_data = stay_data.sort_values('time_step', ascending=True)
            
            # Split data: Pre-ICU (< 0) and Post-ICU (>= 0)
            pre_icu = stay_data[stay_data['time_step'] < 0]
            post_icu = stay_data[stay_data['time_step'] >= 0]
            
            if pre_icu.empty or post_icu.empty:
                continue
                
            # Fill missing values
            pre_icu_filled = self._fill_missing_values(pre_icu)
            post_icu_filled = self._fill_missing_values(post_icu)
            
            if pre_icu_filled is None or post_icu_filled is None:
                continue
                
            try:
                # Generate Knowledge Case
                input_data = self._generate_stay_description(pre_icu_filled)
                output_data = self._generate_output_summary(post_icu_filled)
                
                case = {
                    'stay_id': str(stay_id),
                    'subject_id': input_data['metadata']['subject_id'],
                    'timestamp': datetime.now().isoformat(),
                    'input_description': input_data['description'],
                    'output_summary': output_data['description'],
                    'sofa_scores': input_data['sofa_details'],
                    'sofa_scores_post_icu': output_data['sofa_details']
                }
                
                self.knowledge_base.append(case)
                
            except Exception as e:
                print(f"Error processing stay_id {stay_id}: {e}")
                
        self.save()

    def save(self):
        """Save the constructed knowledge base to JSON."""
        print(f"Saving {len(self.knowledge_base)} cases to {self.output_path}...")
        
        def convert_to_serializable(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            return obj

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2, default=convert_to_serializable)
        print("Knowledge Base saved successfully.")

if __name__ == "__main__":
    # Configuration
    CSV_PATH = 'ai_clinician_dataset.csv'
    OUTPUT_PATH = 'knowledge_base_v1.json'
    
    builder = KnowledgeBaseBuilder(CSV_PATH, OUTPUT_PATH)
    builder.build()
