#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main program: integrates prediction and evaluation with English UI, model dropdown,
auto mode, tab navigation, and computed SOFA charts.
"""

import sys
import json
import os
import threading
import time
import argparse

# Import patient SOFA calculators (with safe fallbacks)
try:
    from patient_text_generator2 import (
        calculate_sofa_respiration,
        calculate_sofa_coagulation,
        calculate_sofa_liver,
        calculate_sofa_cardiovascular,
        calculate_sofa_cns,
        calculate_sofa_renal,
        calculate_total_sofa
    )
except Exception:
    def calculate_sofa_respiration(*args, **kwargs):
        return 0.0
    def calculate_sofa_coagulation(*args, **kwargs):
        return 0.0
    def calculate_sofa_liver(*args, **kwargs):
        return 0.0
    def calculate_sofa_cardiovascular(*args, **kwargs):
        return 0.0
    def calculate_sofa_cns(*args, **kwargs):
        return 0.0
    def calculate_sofa_renal(*args, **kwargs):
        return 0.0
    def calculate_total_sofa(*args, **kwargs):
        return sum([v for v in args if isinstance(v, (int, float))])

# Prediction agent
try:
    from experiment import AdaptiveExperimentAgent, configure_dspy, validate_sofa_features
except Exception:
    AdaptiveExperimentAgent = None
    def configure_dspy(*args, **kwargs):
        print("Warning: DSPy/experiment not available. Please install dependencies (e.g., `pip install dspy-ai`) or activate the provided conda env from environment.yml.")
        return None
    def validate_sofa_features(*args, **kwargs):
        return {'is_dict': False, 'issues': ['validator unavailable']}

# Evaluation helpers
from sofa_prediction_evaluator import (
    evaluate_with_ollama,
    extract_patient_id,
    save_evaluation_report
)

# Input helpers
from sofa_prediction_input import extract_patient_id as input_extract_patient_id

# Optional factual prediction and model list
try:
    from fact_prediction import process_all_fact_predictions, MODEL_NAMES, run_fact_prediction
except Exception:
    process_all_fact_predictions = None
    run_fact_prediction = None
    MODEL_NAMES = [
        "deepseek-r1:32b",
        "gemma3:12b",
        "qwen3:30b"
    ]

# Optional visualization module
try:
    from generate_model_trust_chart import generate_model_trust_chart
    VISUALIZE_AVAILABLE = True
except Exception:
    generate_model_trust_chart = None
    VISUALIZE_AVAILABLE = False

# Import core functions
from core_functions import (
    run_prediction,
    run_evaluation,
    select_best_prediction,
    save_best_prediction_result
)

# Database module for web application
try:
    from database import (
        SessionLocal, get_db,
        Patient, Prediction, PredictionResult, SystemConfig, User,
        PatientCRUD, PredictionCRUD, PredictionResultCRUD, SystemConfigCRUD
    )
    DATABASE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Database module not available: {e}")
    DATABASE_AVAILABLE = False
    SessionLocal = None
    get_db = None

EXPERT_INTERVENTIONS_PATH = "/data/wzx/extracted_expert_interventions.json"
_EXPERT_INTERVENTION_CACHE = None

def _load_expert_intervention_cache(path: str = EXPERT_INTERVENTIONS_PATH):
    global _EXPERT_INTERVENTION_CACHE
    if isinstance(_EXPERT_INTERVENTION_CACHE, dict):
        return _EXPERT_INTERVENTION_CACHE
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        _EXPERT_INTERVENTION_CACHE = {}
        return _EXPERT_INTERVENTION_CACHE

    mapping = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            pinfo = item.get("patient_info") if isinstance(item.get("patient_info"), dict) else {}
            stay_id = pinfo.get("stay_id")
            einfo = item.get("expert_intervention") if isinstance(item.get("expert_intervention"), dict) else {}
            desc = einfo.get("description")
            if stay_id is None or not isinstance(desc, str) or not desc.strip():
                continue
            mapping[str(stay_id)] = desc.strip()
    _EXPERT_INTERVENTION_CACHE = mapping
    return _EXPERT_INTERVENTION_CACHE

def _lookup_expert_intervention(input_description: str, patient_data: dict):
    mapping = _load_expert_intervention_cache()
    stay_id = None
    if isinstance(patient_data, dict):
        stay_id = patient_data.get("stay_id")
    if stay_id is None:
        try:
            stay_id = extract_patient_id(input_description or "", patient_data)
        except Exception:
            stay_id = None
    if stay_id is None:
        return ""
    return mapping.get(str(stay_id), "")


def process_case_auto(model_names, patient_data):
    """Auto mode: pick best result by rotating models"""
    desc = patient_data.get('input_description')
    if not desc:
        raise ValueError('缺少患者描述 input_description')
    
    # 优先使用 patient_data 提取 ID
    pid = extract_patient_id(desc, patient_data)
    
    if not pid or pid == 'unknown':
        raise ValueError('无法从患者描述提取患者编号')
    intervention = patient_data.get('intervention')
    if not intervention:
        expert = _lookup_expert_intervention(desc, patient_data)
        if expert:
            patient_data["intervention"] = expert
            intervention = expert
    if not intervention:
        raise ValueError('缺少干预措施 intervention')
    print(f"Processing case: {pid}")

    best_result = select_best_prediction(model_names, patient_data)

    if best_result:
        patient_id = extract_patient_id(patient_data['input_description'], patient_data)
        output_file = f"./output/best_result/result_{patient_id}.json"
        save_best_prediction_result(best_result, output_file)
        print(f"Case {patient_id} completed")
        return True
    else:
        print("Processing failed")
        return False


def process_all_cases_auto(model_names, data_file="./icu_stays_descriptions88.json"):
    """Auto mode for all cases; prompts for intervention or next"""
    print("Starting auto mode...")

    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            all_patient_data = json.load(f)
    except FileNotFoundError:
        print(f"Data file not found: {data_file}")
        return
    except Exception as e:
        print(f"Error reading data file: {str(e)}")
        return

    for i, patient_data in enumerate(all_patient_data):
        print(f"\nProcessing {i+1}/{len(all_patient_data)}")
        # 传递 patient_data 以支持直接 ID 查找
        patient_id = extract_patient_id(patient_data.get('input_description', ''), patient_data)
        print(f"Patient ID: {patient_id}")
        print(f"Patient info: {patient_data.get('input_description', '')[:100]}...")

        print("\nEnter intervention 或输入 'next' 跳过当前病例:")
        user_input = input().strip()

        if user_input.lower() == 'next':
            print('跳过当前病例')
            continue
        if not user_input:
            expert = _lookup_expert_intervention(patient_data.get("input_description", ""), patient_data)
            if not expert:
                raise ValueError('未输入干预措施，且未找到专家干预方案')
            patient_data["intervention"] = expert
            print(f"Intervention from expert interventions: {expert}")
        else:
            patient_data["intervention"] = user_input
            print(f"Intervention added: {user_input}")

        process_case_auto(model_names, patient_data)
        time.sleep(1)

    print("All cases processed")


def cli_main():
    """Unified CLI: supports single, auto, confidence, and batch modes.
    当无图形环境且未提供参数时，进入交互式向导，逐步采集输入。"""
    parser = argparse.ArgumentParser(description='Sepsis Prediction & Evaluation CLI')
    parser.add_argument('--mode', choices=['single', 'auto', 'confidence', 'batch', 'debug', 'visualize'], default='single',
                        help='运行模式：single（单次预测）、auto（三模型自动轮换）、confidence（事实预测与信任评估）、batch（批量自动处理）、visualize（生成可视化图表）')
    parser.add_argument('--model', type=str, default=None, help='单次预测模型名称，例如 gemma3:12b')
    parser.add_argument('--models', type=str, nargs='*', default=None, help='自动模式的三个模型名称列表')
    parser.add_argument('--input', type=str, default=None, help='患者描述文本')
    parser.add_argument('--intervention', type=str, default=None, help='干预措施文本')
    parser.add_argument('--data-file', type=str, default='./icu_stays_descriptions88.json', help='批量/调试模式输入JSON文件')
    parser.add_argument('--patient-id', type=str, default=None, help='调试模式患者 stay_id（或留空取第一条）')
    # 接受 --cli 作为兼容标志，避免未识别参数错误
    parser.add_argument('--cli', action='store_true', help='使用命令行模式（兼容标志）')
    args = parser.parse_args()

    # 默认模型集合
    default_models = MODEL_NAMES[:3] if len(MODEL_NAMES) >= 3 else MODEL_NAMES

    # 交互式向导：无额外参数或 single/auto 缺少 --input 时触发
    user_args = [a for a in sys.argv[1:] if a not in ('--cli',)]
    if (len(user_args) == 0) or (args.mode in ('single', 'auto') and args.input is None):
        print('进入命令行交互模式。按 Enter 使用默认值。')
        try:
            mode = input('选择运行模式 [single/auto/confidence/batch/visualize] (默认 single): ').strip().lower()
        except EOFError:
            mode = ''
        if not mode:
            mode = 'single'

        if mode == 'single':
            try:
                model = input(f"选择模型 (默认 {MODEL_NAMES[0]}): ").strip()
            except EOFError:
                model = ''
            if not model:
                model = MODEL_NAMES[0]
            desc = ''
            while not desc:
                try:
                    desc = input('请输入患者描述：').strip()
                except EOFError:
                    desc = ''
                if not desc:
                    print('患者描述不能为空，请重新输入。')
            try:
                intervention = input('输入干预措施: ').strip()
            except EOFError:
                intervention = ''
            if not intervention:
                print('错误：需要输入干预措施')
                return 1
            patient_data = {
                'input_description': desc,
                'intervention': intervention,
                'vital_signs': {}
            }
            prediction_file = run_prediction(model, patient_data)
            if prediction_file:
                confidence = run_evaluation(model, prediction_file)
                try:
                    with open(prediction_file, 'r', encoding='utf-8') as f:
                        prediction_data = json.load(f)
                    best_result = {
                        'prediction_model': model,
                        'evaluation_models': [model],
                        'total_confidence': confidence if confidence is not None else 0,
                        'prediction_file': prediction_file,
                        'prediction_data': prediction_data
                    }
                    pid = extract_patient_id(desc)
                    out_file = f"./result_{pid}.json"
                    save_best_prediction_result(best_result, out_file)
                    print(f'完成，结果文件：{out_file}')
                except Exception as e:
                    print(f'生成结果文件失败：{e}')
            else:
                print('预测失败')
            return 0

        elif mode == 'auto':
            try:
                models_str = input(f"输入三个模型，用空格分隔 (默认 {' '.join(default_models[:3])}): ").strip()
            except EOFError:
                models_str = ''
            chosen = models_str.split() if models_str else default_models[:3]
            desc = ''
            while not desc:
                try:
                    desc = input('请输入患者描述：').strip()
                except EOFError:
                    desc = ''
                if not desc:
                    print('患者描述不能为空，请重新输入。')
            try:
                intervention = input('输入干预措施 : ').strip()
            except EOFError:
                intervention = ''
            patient_data = {
                'input_description': desc,
                'intervention': intervention,
                'vital_signs': {}
            }
            if not patient_data.get("intervention"):
                expert = _lookup_expert_intervention(desc, patient_data)
                if expert:
                    patient_data["intervention"] = expert
                    print(f"Auto 模式：使用专家干预方案：{expert}")
                else:
                    print('错误：未输入干预措施，且未找到专家干预方案')
                    return 1
            best_result = select_best_prediction(chosen[:3], patient_data)
            if best_result:
                pid = extract_patient_id(desc)
                out_file = f"./result_{pid}.json"
                save_best_prediction_result(best_result, out_file)
                print(f"完成，最佳模型：{best_result['prediction_model']}，结果文件：{out_file}")
            else:
                print('自动轮换失败')
            return 0

        elif mode == 'confidence':
            if process_all_fact_predictions:
                try:
                    process_all_fact_predictions()
                    print('完成事实预测与信任评估')
                except Exception as e:
                    print(f'事实预测与信任评估失败: {e}')
                    return 1
            else:
                print('事实预测模块不可用，已跳过')
            return 0

        elif mode == 'batch':
            try:
                df = input(f"输入数据文件路径 (默认 {args.data_file}): ").strip()
            except EOFError:
                df = ''
            if not df:
                df = args.data_file
            try:
                mods = input(f"三个模型，用空格分隔 (默认 {' '.join(default_models[:3])}): ").strip()
            except EOFError:
                mods = ''
            chosen = mods.split() if mods else default_models[:3]
            try:
                intervention = input('输入干预措施 : ').strip()
            except EOFError:
                intervention = ''
            try:
                with open(df, 'r', encoding='utf-8') as f:
                    all_patient_data = json.load(f)
            except Exception as e:
                print(f'读取数据文件失败：{e}')
                return 1
            print(f'批量自动处理，共 {len(all_patient_data)} 个病例')
            for i, patient_data in enumerate(all_patient_data, 1):
                desc = patient_data.get('input_description')
                if not desc:
                    print(f'[{i}] 缺少输入描述，跳过')
                    continue
                if intervention:
                    patient_data['intervention'] = intervention
                else:
                    expert = _lookup_expert_intervention(desc, patient_data)
                    if not expert:
                        print(f'[{i}] 未提供干预措施，且未找到专家干预方案，跳过')
                        continue
                    patient_data['intervention'] = expert
                patient_data.setdefault('vital_signs', {})
                print(f'[{i}] 处理患者 {extract_patient_id(desc)}')
                process_case_auto(chosen[:3], patient_data)
                time.sleep(0.5)
            print('批量处理完成')
            return 0

        elif mode == 'visualize':
            if not VISUALIZE_AVAILABLE or generate_model_trust_chart is None:
                print('可视化模块不可用')
                print('请确保generate_model_trust_chart.py文件存在且依赖项已安装')
                return 1

            print('开始生成模型信任分析图表...')
            try:
                success = generate_model_trust_chart()
                if success:
                    print('✅ 可视化图表生成完成！')
                else:
                    print('❌ 图表生成失败！')
                    return 1
            except Exception as e:
                print(f'生成图表时出错: {e}')
                import traceback
                traceback.print_exc()
                return 1
            return 0

        else:
            print('未知模式')
            return 1

    # 非交互路径：按传入参数执行
    if args.mode == 'single':
        if not args.model:
            args.model = MODEL_NAMES[0]
        if not args.input:
            print('错误：single 模式需要 --input 患者描述')
            return 1
        patient_data = {
            'input_description': args.input,
            'intervention': args.intervention,
            'vital_signs': {}
        }
        prediction_file = run_prediction(args.model, patient_data)
        if prediction_file:
            confidence = run_evaluation(args.model, prediction_file)
            try:
                with open(prediction_file, 'r', encoding='utf-8') as f:
                    prediction_data = json.load(f)
                best_result = {
                    'prediction_model': args.model,
                    'evaluation_models': [args.model],
                    'total_confidence': confidence if confidence is not None else 0,
                    'prediction_file': prediction_file,
                    'prediction_data': prediction_data
                }
                pid = extract_patient_id(args.input, patient_data)
                out_file = f"./result_{pid}.json"
                save_best_prediction_result(best_result, out_file)
                print(f'完成，结果文件：{out_file}')
            except Exception as e:
                print(f'生成结果文件失败：{e}')
        else:
            print('预测失败')
        return 0

    elif args.mode == 'auto':
        chosen = args.models if args.models else default_models
        if args.input is None:
            print('错误：auto 模式需要 --input 患者描述')
            return 1
        patient_data = {
            'input_description': args.input,
            'intervention': args.intervention,
            'vital_signs': {}
        }
        if not patient_data.get("intervention"):
            expert = _lookup_expert_intervention(args.input, patient_data)
            if expert:
                patient_data["intervention"] = expert
            else:
                print('错误：auto 模式未提供 --intervention，且未找到专家干预方案')
                return 1
        best_result = select_best_prediction(chosen[:3], patient_data)
        if best_result:
            pid = extract_patient_id(args.input, patient_data)
            out_file = f"./result_{pid}.json"
            save_best_prediction_result(best_result, out_file)
            print(f'完成，最佳模型：{best_result["prediction_model"]}，结果文件：{out_file}')
        else:
            print('自动轮换失败')
        return 0

    elif args.mode == 'confidence':
        if process_all_fact_predictions:
            try:
                process_all_fact_predictions()
                print('完成事实预测与信任评估')
            except Exception as e:
                print(f'事实预测与信任评估失败: {e}')
                return 1
        else:
            print('事实预测模块不可用，已跳过')
        return 0

    elif args.mode == 'batch':
        # 非交互批量自动处理：对 data-file 中每个病例应用三模型自动轮换
        chosen = args.models if args.models else default_models
        try:
            with open(args.data_file, 'r', encoding='utf-8') as f:
                all_patient_data = json.load(f)
        except Exception as e:
            print(f'读取数据文件失败：{e}')
            return 1
        print(f'批量自动处理，共 {len(all_patient_data)} 个病例')
        for i, patient_data in enumerate(all_patient_data, 1):
            desc = patient_data.get('input_description')
            if not desc:
                print(f'[{i}] 缺少输入描述，跳过')
                continue
            if args.intervention:
                patient_data['intervention'] = args.intervention
            else:
                expert = _lookup_expert_intervention(desc, patient_data)
                if not expert:
                    print(f'[{i}] 未提供 --intervention，且未找到专家干预方案，跳过')
                    continue
                patient_data['intervention'] = expert
            patient_data.setdefault('vital_signs', {})
            pid = extract_patient_id(desc)
            if not pid or pid == 'unknown':
                print(f'[{i}] 无法从描述提取患者编号，已终止')
                return 1
            print(f'[{i}] 处理患者 {pid}')
            process_case_auto(chosen[:3], patient_data)
            time.sleep(0.5)
        print('批量处理完成')
        return 0

    elif args.mode == 'debug':
        try:
            with open(args.data_file, 'r', encoding='utf-8') as f:
                all_patient_data = json.load(f)
        except Exception as e:
            print(f'读取数据文件失败：{e}')
            return 1
        target = None
        if args.patient_id:
            for entry in all_patient_data:
                if str(entry.get('stay_id')) == str(args.patient_id):
                    target = entry
                    break
        if not target:
            target = all_patient_data[0] if all_patient_data else None
        if not target:
            print('数据为空')
            return 1
        desc = target.get('input_description', '')
        if not desc:
            print('缺少患者描述')
            return 1
        if not target.get('intervention'):
            print('调试模式：缺少干预措施 intervention')
            return 1
        target.setdefault('vital_signs', {})
        chosen = args.models if args.models else MODEL_NAMES[:3]
        print('Step 1: 生成各模型预测')
        best_result = select_best_prediction(chosen[:3], target)
        if not best_result:
            print('选择最佳预测失败')
            return 1
        print('Step 2: 评估预测并选择前三评估模型')
        print(f"selected_evaluators: {best_result.get('selected_evaluators', [])}")
        print(f"evaluator_scores: {best_result.get('evaluator_scores', {})}")
        print('Step 3: 使用前五模型进行干预分析与风险评分')
        for r in best_result.get('per_model_risk', []):
            print(f"  模型 {r['model']} -> 风险等级 {r['risk_level']}，评分 {r['risk_score']}")
        print(f"最终加权风险评分: {best_result.get('final_weighted_risk_score')}")
        print(f"最终加权风险等级: {best_result.get('final_weighted_risk_level')}")
        pid = extract_patient_id(desc)
        out_file = f"./result_{pid}.json"
        save_best_prediction_result(best_result, out_file)
        print(f'完成，结果文件：{out_file}')
        return 0

    elif args.mode == 'visualize':
        if not VISUALIZE_AVAILABLE or generate_model_trust_chart is None:
            print('可视化模块不可用')
            print('请确保generate_model_trust_chart.py文件存在且依赖项已安装')
            return 1

        print('开始生成模型信任分析图表...')
        try:
            success = generate_model_trust_chart()
            if success:
                print('✅ 可视化图表生成完成！')
            else:
                print('❌ 图表生成失败！')
                return 1
        except Exception as e:
            print(f'生成图表时出错: {e}')
            import traceback
            traceback.print_exc()
            return 1
        return 0

    else:
        print('未知模式')
        return 1


def start_web_ui(port: int = 8000):
    try:
        import http.server
        import socketserver
        from pathlib import Path
        from urllib.parse import urlparse
        try:
            os.chdir(str(Path(__file__).resolve().parent))
        except Exception:
            pass

        AUTO_DATA_FILE = "./icu_stays_descriptions88.json"

        class AppHandler(http.server.SimpleHTTPRequestHandler):
            server_version = "SepsisWeb/1.0"
            protocol_version = "HTTP/1.1"

            def _send_json(self, obj, status=200):
                data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _bad(self, msg, status=400):
                self._send_json({"error": str(msg)}, status)

            def do_GET(self):
                if self.path.startswith("/api/"):
                    if self.path.startswith("/api/models"):
                        return self._send_json({"models": MODEL_NAMES})
                    if self.path.startswith("/api/next_auto_case"):
                        try:
                            cases = getattr(self.server, "auto_cases", None)
                            idx = getattr(self.server, "auto_index", 0)
                            if not cases:
                                with open(AUTO_DATA_FILE, "r", encoding="utf-8") as f:
                                    cases = json.load(f)
                                self.server.auto_cases = cases
                                self.server.auto_index = 0
                                idx = 0
                            if idx >= len(cases):
                                return self._bad("no_more_cases", 404)
                            case = cases[idx]
                            try:
                                self.server.current_case = case
                            except Exception:
                                pass
                            self.server.auto_index = idx + 1
                            return self._send_json({
                                "stay_id": case.get("stay_id"),
                                "input_description": case.get("input_description", "")
                            })
                        except FileNotFoundError:
                            return self._bad("auto_data_file_not_found", 404)
                        except Exception as e:
                            return self._bad(str(e), 500)
                    if self.path.startswith("/api/ping"):
                        return self._send_json({"status": "ok"})

                    # New database API endpoints
                    if self.path.startswith("/api/patients/list"):
                        try:
                            if not DATABASE_AVAILABLE:
                                return self._bad("database_not_available", 503)
                            db = SessionLocal()
                            try:
                                skip = int(self.path.split("?skip=")[1].split("&")[0]) if "?skip=" in self.path else 0
                                limit = int(self.path.split("?limit=")[1].split("&")[0]) if "?limit=" in self.path else 100
                            except:
                                skip = 0
                                limit = 100
                            patients = PatientCRUD.get_all(db, skip=skip, limit=limit)
                            total = db.query(Patient).count()
                            result = {
                                "patients": [
                                    {
                                        "id": p.id,
                                        "stay_id": p.stay_id,
                                        "subject_id": p.subject_id,
                                        "age": p.age,
                                        "gender": p.gender,
                                        "input_description": p.input_description[:200] + "..." if p.input_description and len(p.input_description) > 200 else p.input_description
                                    }
                                    for p in patients
                                ],
                                "total": total,
                                "skip": skip,
                                "limit": limit
                            }
                            db.close()
                            return self._send_json(result)
                        except Exception as e:
                            if 'db' in locals():
                                db.close()
                            return self._bad(str(e), 500)

                    if self.path.startswith("/api/patients/"):
                        try:
                            if not DATABASE_AVAILABLE:
                                return self._bad("database_not_available", 503)
                            # Extract patient ID from path
                            path_parts = self.path.split("/")
                            patient_id_str = path_parts[3]  # /api/patients/{id}
                            if patient_id_str.isdigit():
                                patient_id = int(patient_id_str)
                                db = SessionLocal()
                                patient = PatientCRUD.get_by_id(db, patient_id)
                                db.close()
                                if patient:
                                    return self._send_json({
                                        "id": patient.id,
                                        "stay_id": patient.stay_id,
                                        "subject_id": patient.subject_id,
                                        "age": patient.age,
                                        "gender": patient.gender,
                                        "weight": patient.weight,
                                        "input_description": patient.input_description,
                                        "output_summary": patient.output_summary,
                                        "vital_signs": patient.vital_signs,
                                        "sofa_scores": patient.sofa_scores,
                                        "sofa_scores_post_icu": patient.sofa_scores_post_icu,
                                        "created_at": patient.created_at.isoformat() if patient.created_at else None,
                                        "updated_at": patient.updated_at.isoformat() if patient.updated_at else None
                                    })
                                else:
                                    return self._bad("patient_not_found", 404)
                            else:
                                # Try to get by stay_id
                                db = SessionLocal()
                                patient = PatientCRUD.get_by_stay_id(db, patient_id_str)
                                db.close()
                                if patient:
                                    return self._send_json({
                                        "id": patient.id,
                                        "stay_id": patient.stay_id,
                                        "subject_id": patient.subject_id,
                                        "age": patient.age,
                                        "gender": patient.gender,
                                        "weight": patient.weight,
                                        "input_description": patient.input_description,
                                        "output_summary": patient.output_summary,
                                        "vital_signs": patient.vital_signs,
                                        "sofa_scores": patient.sofa_scores,
                                        "sofa_scores_post_icu": patient.sofa_scores_post_icu,
                                        "created_at": patient.created_at.isoformat() if patient.created_at else None,
                                        "updated_at": patient.updated_at.isoformat() if patient.updated_at else None
                                    })
                                else:
                                    return self._bad("patient_not_found", 404)
                        except Exception as e:
                            if 'db' in locals():
                                db.close()
                            return self._bad(str(e), 500)

                    if self.path.startswith("/api/predictions/list"):
                        try:
                            if not DATABASE_AVAILABLE:
                                return self._send_json({"predictions": [], "total": 0, "skip": 0, "limit": 100})
                            db = SessionLocal()
                            try:
                                skip = int(self.path.split("?skip=")[1].split("&")[0]) if "?skip=" in self.path else 0
                                limit = int(self.path.split("?limit=")[1].split("&")[0]) if "?limit=" in self.path else 100
                            except:
                                skip = 0
                                limit = 100
                            predictions = PredictionCRUD.get_all(db, skip=skip, limit=limit)
                            total = db.query(Prediction).count()
                            result = {
                                "predictions": [
                                    {
                                        "id": p.id,
                                        "task_id": p.task_id,
                                        "patient_id": p.patient_id,
                                        "model_names": p.model_names,
                                        "intervention": p.intervention[:100] + "..." if p.intervention and len(p.intervention) > 100 else p.intervention,
                                        "prediction_mode": p.prediction_mode,
                                        "status": p.status,
                                        "progress": p.progress,
                                        "created_at": p.created_at.isoformat() if p.created_at else None,
                                        "completed_at": p.completed_at.isoformat() if p.completed_at else None
                                    }
                                    for p in predictions
                                ],
                                "total": total,
                                "skip": skip,
                                "limit": limit
                            }
                            db.close()
                            return self._send_json(result)
                        except Exception as e:
                            if 'db' in locals():
                                db.close()
                            return self._bad(str(e), 500)

                    if self.path.startswith("/api/system/config"):
                        try:
                            if not DATABASE_AVAILABLE:
                                return self._send_json({"configs": {}})
                            db = SessionLocal()
                            configs = db.query(SystemConfig).all()
                            config_dict = {
                                config.config_key: config.config_value
                                for config in configs
                            }
                            db.close()
                            return self._send_json({"configs": config_dict})
                        except Exception as e:
                            if 'db' in locals():
                                db.close()
                            return self._bad(str(e), 500)

                    if self.path.startswith("/api/reset"):
                        try:
                            try:
                                self.server.auto_index = 0
                            except Exception:
                                pass
                            try:
                                self.server.current_case = None
                            except Exception:
                                pass
                            return self._send_json({"status": "ok"})
                        except Exception as e:
                            return self._bad(str(e), 500)
                return http.server.SimpleHTTPRequestHandler.do_GET(self)

            def do_POST(self):
                if not self.path.startswith("/api/"):
                    return self._bad("not_found", 404)
                try:
                    length = int(self.headers.get("Content-Length") or "0")
                except Exception:
                    length = 0
                try:
                    body = self.rfile.read(length) if length > 0 else b"{}"
                    payload = json.loads(body.decode("utf-8") or "{}")
                except Exception:
                    return self._bad("invalid_json", 400)

                def _lookup_output_summary(desc_text: str):
                    try:
                        pid = extract_patient_id(desc_text)
                        if not pid or pid == "unknown":
                            return None
                        cases = getattr(self.server, "auto_cases", None)
                        if not cases:
                            try:
                                with open(AUTO_DATA_FILE, "r", encoding="utf-8") as f:
                                    cases = json.load(f)
                                self.server.auto_cases = cases
                            except Exception:
                                cases = None
                        if isinstance(cases, list):
                            for c in cases:
                                try:
                                    if str(c.get("stay_id")) == str(pid):
                                        return c.get("output_summary")
                                except Exception:
                                    continue
                        cur = getattr(self.server, "current_case", None)
                        if isinstance(cur, dict) and str(cur.get("stay_id")) == str(pid):
                            return cur.get("output_summary")
                        return None
                    except Exception:
                        return None

                def _lookup_baseline_sofa_totals(desc_text: str):
                    try:
                        # 尝试从描述中提取ID
                        pid = extract_patient_id(desc_text)
                        
                        # 尝试查找对应的完整数据对象
                        cases = getattr(self.server, "auto_cases", None)
                        if not cases:
                            try:
                                with open(AUTO_DATA_FILE, "r", encoding="utf-8") as f:
                                    cases = json.load(f)
                                self.server.auto_cases = cases
                            except Exception:
                                cases = None
                        
                        target_case = None
                        if isinstance(cases, list):
                            for c in cases:
                                try:
                                    # 尝试匹配 stay_id 或 subject_id
                                    c_pid = str(c.get("stay_id"))
                                    c_sub_id = str(c.get("subject_id"))
                                    if c_pid == str(pid) or c_sub_id == str(pid):
                                        target_case = c
                                        break
                                    
                                    # 如果pid是从input_description提取的，也尝试反向匹配
                                    if str(extract_patient_id(c.get("input_description", ""), c)) == str(pid):
                                        target_case = c
                                        break
                                except Exception:
                                    continue
                        
                        if target_case:
                            seq = target_case.get("sofa_scores") or target_case.get("sofa_scores_post_icu")
                            if isinstance(seq, list) and seq:
                                totals = {}
                                for i, item in enumerate(seq):
                                    try:
                                        totals[str(i)] = int(item.get("sofa_total", 0))
                                    except Exception:
                                        pass
                                return totals
                        return None
                    except Exception:
                        return None

                def _safe_model_name(name: str):
                    try:
                        return (name or "").split(":")[0]
                    except Exception:
                        return name

                def _find_fact_files(model_name: str):
                    try:
                        import os
                        safe = _safe_model_name(model_name)
                        base = f"./output/{safe}"
                        if not os.path.isdir(base):
                            return []
                        files = []
                        for fn in os.listdir(base):
                            if fn.startswith("fact_prediction_result_") and fn.endswith(f"_{model_name}.json"):
                                files.append(os.path.join(base, fn))
                        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                        return files
                    except Exception:
                        return []

                def _load_json(path: str):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            return json.load(f)
                    except Exception:
                        return None

                def _binary_metrics(pred_labels, pred_probs, true_labels):
                    try:
                        n = len(true_labels)
                        if n == 0:
                            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}
                        tp = 0; fp = 0; tn = 0; fn = 0
                        for i in range(n):
                            y = 1 if true_labels[i] else 0
                            yhat = 1 if pred_labels[i] else 0
                            if yhat == 1 and y == 1:
                                tp += 1
                            elif yhat == 1 and y == 0:
                                fp += 1
                            elif yhat == 0 and y == 0:
                                tn += 1
                            else:
                                fn += 1
                        acc = (tp + tn) / n if n else 0.0
                        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                        probs = list(pred_probs) if isinstance(pred_probs, list) else []
                        uniq = sorted(set([min(max(float(p), 0.0), 1.0) for p in probs] + [0.0, 1.0]))
                        roc = []
                        pos = sum(1 for y in true_labels if y)
                        neg = n - pos
                        for t in uniq:
                            tp2 = 0; fp2 = 0; tn2 = 0; fn2 = 0
                            for i in range(n):
                                y = 1 if true_labels[i] else 0
                                yhatp = 1 if probs[i] >= t else 0
                                if yhatp == 1 and y == 1:
                                    tp2 += 1
                                elif yhatp == 1 and y == 0:
                                    fp2 += 1
                                elif yhatp == 0 and y == 0:
                                    tn2 += 1
                                else:
                                    fn2 += 1
                            tpr = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0.0
                            fpr = fp2 / (fp2 + tn2) if (fp2 + tn2) > 0 else 0.0
                            roc.append((fpr, tpr))
                        roc.sort(key=lambda x: x[0])
                        auc = 0.0
                        for i in range(1, len(roc)):
                            x0, y0 = roc[i-1]
                            x1, y1 = roc[i]
                            auc += (x1 - x0) * (y0 + y1) * 0.5
                        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}
                    except Exception:
                        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}

                def _compute_confidence_metrics(report: dict):
                    comps = ["sofa_respiration","sofa_coagulation","sofa_liver","sofa_cardiovascular","sofa_cns","sofa_renal"]
                    ps = report.get("predicted_sofa_scores", {})
                    ascores = report.get("actual_sofa_scores", {})
                    per = {}
                    all_pred = []
                    all_true = []
                    all_prob = []
                    for k in comps:
                        p = ps.get(k) or []
                        a = ascores.get(k) or []
                        m = min(len(p), len(a))
                        p = list(p)[:m]
                        a = list(a)[:m]
                        pred_labels = [(1 if (int(p[i]) >= 2) else 0) for i in range(m)]
                        true_labels = [(1 if (int(a[i]) >= 2) else 0) for i in range(m)]
                        pred_probs = [max(0.0, min(1.0, (float(p[i]) / 4.0))) for i in range(m)]
                        met = _binary_metrics(pred_labels, pred_probs, true_labels)
                        per[k] = met
                        all_pred.extend(pred_labels)
                        all_true.extend(true_labels)
                        all_prob.extend(pred_probs)
                    overall = _binary_metrics(all_pred, all_prob, all_true)
                    trust = float(report.get("model_trust_score", 0.0) or 0.0)
                    return {"overall": overall, "components": per, "trust": trust}

                # New database POST endpoints
                if self.path.startswith("/api/patients/create"):
                    try:
                        if not DATABASE_AVAILABLE:
                            return self._bad("database_not_available", 503)

                        required_fields = ["stay_id", "input_description"]
                        for field in required_fields:
                            if field not in payload:
                                return self._bad(f"missing_{field}", 400)

                        db = SessionLocal()

                        # Check if patient already exists
                        existing = PatientCRUD.get_by_stay_id(db, payload["stay_id"])
                        if existing:
                            db.close()
                            return self._bad("patient_already_exists", 409)

                        # Create patient
                        patient_data = {
                            "stay_id": payload["stay_id"],
                            "subject_id": payload.get("subject_id", ""),
                            "input_description": payload["input_description"],
                            "output_summary": payload.get("output_summary", ""),
                            "age": payload.get("age"),
                            "gender": payload.get("gender"),
                            "weight": payload.get("weight"),
                            "vital_signs": payload.get("vital_signs"),
                            "sofa_scores": payload.get("sofa_scores"),
                            "sofa_scores_post_icu": payload.get("sofa_scores_post_icu"),
                            "source_file": payload.get("source_file", "api"),
                            "notes": payload.get("notes", "")
                        }

                        patient = PatientCRUD.create(db, patient_data)
                        db.close()

                        return self._send_json({
                            "status": "success",
                            "patient_id": patient.id,
                            "message": "Patient created successfully"
                        })

                    except Exception as e:
                        if 'db' in locals():
                            db.close()
                        return self._bad(str(e), 500)

                if self.path.startswith("/api/predictions/create"):
                    try:
                        if not DATABASE_AVAILABLE:
                            return self._bad("database_not_available", 503)

                        required_fields = ["patient_id", "intervention", "model_names"]
                        for field in required_fields:
                            if field not in payload:
                                return self._bad(f"missing_{field}", 400)

                        db = SessionLocal()

                        # Verify patient exists
                        patient = PatientCRUD.get_by_id(db, payload["patient_id"])
                        if not patient:
                            db.close()
                            return self._bad("patient_not_found", 404)

                        # Create prediction task
                        prediction_data = {
                            "patient_id": payload["patient_id"],
                            "model_names": payload["model_names"],
                            "intervention": payload["intervention"],
                            "prediction_mode": payload.get("prediction_mode", "single"),
                            "prediction_parameters": payload.get("prediction_parameters", {}),
                            "status": "pending",
                            "progress": 0.0
                        }

                        # Generate task ID (for Celery integration later)
                        import uuid
                        prediction_data["task_id"] = f"task_{uuid.uuid4().hex[:8]}"

                        prediction = PredictionCRUD.create(db, prediction_data)
                        db.close()

                        return self._send_json({
                            "status": "success",
                            "prediction_id": prediction.id,
                            "task_id": prediction.task_id,
                            "message": "Prediction task created successfully"
                        })

                    except Exception as e:
                        if 'db' in locals():
                            db.close()
                        return self._bad(str(e), 500)

                if self.path.startswith("/api/predictions/update"):
                    try:
                        if not DATABASE_AVAILABLE:
                            return self._bad("database_not_available", 503)

                        required_fields = ["prediction_id", "status"]
                        for field in required_fields:
                            if field not in payload:
                                return self._bad(f"missing_{field}", 400)

                        db = SessionLocal()

                        # Update prediction status
                        prediction = PredictionCRUD.update_status(
                            db,
                            payload["prediction_id"],
                            payload["status"],
                            progress=payload.get("progress"),
                            error=payload.get("error_message")
                        )

                        if not prediction:
                            db.close()
                            return self._bad("prediction_not_found", 404)

                        db.close()

                        return self._send_json({
                            "status": "success",
                            "prediction_id": prediction.id,
                            "task_status": prediction.status,
                            "progress": prediction.progress,
                            "message": "Prediction status updated successfully"
                        })

                    except Exception as e:
                        if 'db' in locals():
                            db.close()
                        return self._bad(str(e), 500)

                if self.path.startswith("/api/run_single"):
                    try:
                        model = payload.get("model") or (MODEL_NAMES[0] if MODEL_NAMES else None)
                        desc = payload.get("input_description") or ""
                        inv = payload.get("intervention") or ""
                        output_summary = payload.get("output_summary")
                        if not model:
                            return self._bad("missing_model", 400)
                        if not desc:
                            return self._bad("missing_input_description", 400)
                        if not inv:
                            return self._bad("missing_intervention", 400)
                        patient_data = {
                            "input_description": desc,
                            "intervention": inv,
                            "vital_signs": {},
                            "output_summary": (output_summary if output_summary is not None else (_lookup_output_summary(desc) or {}))
                        }
                        prediction_file = run_prediction(model, patient_data)
                        if not prediction_file:
                            return self._bad("prediction_failed", 500)
                        with open(prediction_file, "r", encoding="utf-8") as f:
                            prediction_data = json.load(f)
                        conf = run_evaluation(model, prediction_file)
                        baseline_totals = _lookup_baseline_sofa_totals(desc) or None
                        best_result = {
                            "prediction_model": model,
                            "evaluation_models": [model],
                            "total_confidence": conf if conf is not None else 0,
                            "prediction_file": prediction_file,
                            "prediction_data": prediction_data,
                            "output_summary": patient_data.get("output_summary"),
                            "baseline_sofa_totals": baseline_totals
                        }
                        pid = extract_patient_id(desc, patient_data)
                        out_file = f"./result_{pid}.json"
                        save_best_prediction_result(best_result, out_file)
                        return self._send_json(best_result)
                    except Exception as e:
                        return self._bad(str(e), 500)

                if self.path.startswith("/api/run_auto"):
                    try:
                        models = payload.get("models") or (MODEL_NAMES[:3] if len(MODEL_NAMES) >= 3 else MODEL_NAMES)
                        desc = payload.get("input_description") or ""
                        inv = payload.get("intervention") or ""
                        output_summary = payload.get("output_summary")
                        if not desc:
                            return self._bad("missing_input_description", 400)
                        if not inv:
                            expert = _lookup_expert_intervention(desc, payload if isinstance(payload, dict) else {})
                            if expert:
                                inv = expert
                            else:
                                return self._bad("missing_intervention", 400)
                        patient_data = {
                            "input_description": desc,
                            "intervention": inv,
                            "vital_signs": {},
                            "output_summary": (output_summary if output_summary is not None else (_lookup_output_summary(desc) or {}))
                        }
                        best_result = select_best_prediction(models[:3], patient_data)
                        if not best_result:
                            try:
                                fallback_model = models[0] if models else (MODEL_NAMES[0] if MODEL_NAMES else None)
                                if not fallback_model:
                                    return self._bad("auto_failed", 500)
                                pf = run_prediction(fallback_model, patient_data)
                                if not pf:
                                    return self._bad("auto_failed", 500)
                                with open(pf, "r", encoding="utf-8") as f:
                                    pd = json.load(f)
                                conf = run_evaluation(fallback_model, pf)
                                baseline_totals = _lookup_baseline_sofa_totals(desc) or None
                                best_result = {
                                    "prediction_model": fallback_model,
                                    "evaluation_models": [fallback_model],
                                    "total_confidence": conf if isinstance(conf, (int, float)) else 0,
                                    "prediction_file": pf,
                                    "prediction_data": pd,
                                    "output_summary": patient_data.get("output_summary"),
                                    "baseline_sofa_totals": baseline_totals
                                }
                            except Exception:
                                return self._bad("auto_failed", 500)
                        pid = extract_patient_id(desc, patient_data)
                        try:
                            baseline_totals = _lookup_baseline_sofa_totals(desc) or None
                            if isinstance(best_result, dict):
                                best_result["baseline_sofa_totals"] = baseline_totals
                        except Exception:
                            pass
                        out_file = f"./output/best_result/result_{pid}.json"
                        save_best_prediction_result(best_result, out_file)
                        return self._send_json(best_result)
                    except Exception as e:
                        return self._bad(str(e), 500)

                if self.path.startswith("/api/confidence"):
                    try:
                        model = payload.get("model")
                        data_file = payload.get("data_file") or AUTO_DATA_FILE
                        if model:
                            files = _find_fact_files(model)
                            if not files and run_fact_prediction:
                                try:
                                    with open(data_file, "r", encoding="utf-8") as f:
                                        all_patient_data = json.load(f)
                                except Exception as e:
                                    return self._bad(f"read_data_failed: {e}", 500)
                                for item in all_patient_data:
                                    try:
                                        run_fact_prediction(item, model)
                                    except Exception:
                                        continue
                                files = _find_fact_files(model)
                            if not files:
                                return self._send_json({"status": "no_result", "model": model})
                            path = files[0]
                            rep = _load_json(path)
                            if not isinstance(rep, dict):
                                return self._bad("invalid_report", 500)
                            metrics = _compute_confidence_metrics(rep)
                            pid = rep.get("patient_id") or "unknown"
                            return self._send_json({
                                "status": "ok",
                                "mode": "confidence",
                                "model": model,
                                "patient_id": pid,
                                "file": path,
                                "model_trust_score": metrics.get("trust", 0.0),
                                "metrics": metrics
                            })
                        elif process_all_fact_predictions:
                            process_all_fact_predictions(data_file)
                            return self._send_json({"status": "ok", "mode": "confidence", "models": MODEL_NAMES})
                        else:
                            return self._send_json({"status": "skipped"})
                    except Exception as e:
                        return self._bad(str(e), 500)

                return self._bad("not_found", 404)

        try:
            httpd = socketserver.ThreadingTCPServer(("", port), AppHandler)
        except OSError:
            httpd = socketserver.ThreadingTCPServer(("", 0), AppHandler)
            port = httpd.server_address[1]
        print(f"Web GUI 已启动：请打开 http://0.0.0.0:{port}/index_new.html")
        print(f"传统界面：http://0.0.0.0:{port}/ui_preview.html")
        print(f"新版医疗专业界面已就绪，提供现代化用户体验和完整功能")
        print("提示：按 Ctrl+C 结束；若端口不可访问可设置环境变量 WEB_PORT 或使用 --web-port 指定。")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("已停止 Web GUI 服务。")
        finally:
            try:
                httpd.server_close()
            except Exception:
                pass
    except Exception as e:
        print(f"Web GUI 启动失败：{e}")
        print("回退到命令行模式。")
        return cli_main()


def main():
    """Start GUI or fall back to CLI/web depending on environment"""
    args_list = sys.argv[1:]
    force_cli = '--cli' in args_list
    force_web = '--web' in args_list
    show_help = '--help' in args_list or '-h' in args_list

    # 如果有命令行参数（除了--web-port），使用CLI模式
    has_cli_args = any(arg for arg in args_list if not arg.startswith('--web-port'))
    
    # 解析 --web-port（支持 --web-port=8000 或 --web-port 8000）
    port = int(os.environ.get('WEB_PORT', '8000') or 8000)
    for i, a in enumerate(args_list):
        if a.startswith('--web-port='):
            try:
                port = int(a.split('=', 1)[1])
            except Exception:
                pass
        elif a == '--web-port' and i + 1 < len(args_list):
            try:
                port = int(args_list[i + 1])
            except Exception:
                pass

    # 强制网页模式
    if force_web:
        return start_web_ui(port)

    # 如果有CLI参数或强制CLI模式，使用命令行模式
    if force_cli or has_cli_args:
        exit_code = cli_main()
        if isinstance(exit_code, int):
            sys.exit(exit_code)
        return

    # 默认尝试 GUI；若失败则自动切换到网页模式
    try:
        from gui import start_gui
        start_gui()
        return
    except ImportError:
        print('GUI 模块不可用，切换到网页界面模式')
        return start_web_ui(port)
    except Exception as e:
        print(f'GUI 启动失败：{e}，切换到网页界面模式')
        return start_web_ui(port)


if __name__ == "__main__":
    main()
