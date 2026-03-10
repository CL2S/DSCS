#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI module for Septic Shock Prediction & Evaluation
Separated from main.py for better code organization
"""

import sys
import json
import os
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configure matplotlib (Chinese fonts still allowed, minus sign fix)
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# Import new visualization module
try:
    from viz import ChartBuilder, create_sofa_trend_chart, create_sofa_component_chart, create_confidence_chart
    VIZ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization module not available: {e}")
    VIZ_AVAILABLE = False

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

# Evaluation helpers
from sofa_prediction_evaluator import (
    extract_patient_id,
)

# Optional factual prediction and model list
try:
    from fact_prediction import process_all_fact_predictions, MODEL_NAMES
except Exception:
    process_all_fact_predictions = None
    MODEL_NAMES = [
        "gemma3:12b",
        "mistral:7b",
        "qwen3:4b",
        "qwen3:30b",
        "deepseek-r1:32b",
        "medllama2:latest"
    ]

# Import core functions from core_functions module
from core_functions import (
    run_prediction,
    run_evaluation,
    select_best_prediction,
    save_best_prediction_result
)
# Import remaining functions from main module
 

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

def _lookup_expert_intervention(input_description: str, case_data: dict = None):
    mapping = _load_expert_intervention_cache()
    stay_id = None
    if isinstance(case_data, dict):
        stay_id = case_data.get("stay_id")
    if stay_id is None:
        try:
            stay_id = extract_patient_id(input_description or "", case_data)
        except Exception:
            stay_id = None
    if stay_id is None:
        return ""
    return mapping.get(str(stay_id), "")


class PredictionApp:
    """Prediction and Evaluation GUI Application"""
    def __init__(self, root):
        self.root = root
        self.root.title("Septic Shock Prediction & Evaluation")
        self.root.geometry("1200x800")

        # Styles
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("SimHei", 10))
        self.style.configure("TButton", font=("SimHei", 10))
        self.style.configure("TCombobox", font=("SimHei", 10))

        # Frames
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.control_frame = ttk.LabelFrame(self.main_frame, text="Control Panel", padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Mode selection
        ttk.Label(self.control_frame, text="Run Mode:").pack(anchor=tk.W, pady=(0, 5))
        self.mode_var = tk.StringVar(value="single")
        modes = ["single", "auto", "confidence"]
        self.mode_combo = ttk.Combobox(self.control_frame, textvariable=self.mode_var, values=modes, state="readonly")
        self.mode_combo.pack(fill=tk.X, pady=(0, 10))

        # Model selection (single vs auto)
        self.model_single_frame = ttk.Frame(self.control_frame)
        ttk.Label(self.model_single_frame, text="Model:").pack(anchor=tk.W, pady=(0, 5))
        self.model_var = tk.StringVar(value=(MODEL_NAMES[0] if MODEL_NAMES else "gemma3:12b"))
        self.model_combo = ttk.Combobox(self.model_single_frame, textvariable=self.model_var, values=MODEL_NAMES, state="readonly")
        self.model_combo.pack(fill=tk.X, pady=(0, 10))
        self.model_single_frame.pack(fill=tk.X)

        self.model_auto_frame = ttk.Frame(self.control_frame)
        ttk.Label(self.model_auto_frame, text="Auto Mode Models:").pack(anchor=tk.W, pady=(0, 5))
        self.auto_model_vars = [
            tk.StringVar(value=(MODEL_NAMES[0] if MODEL_NAMES else "gemma3:12b")),
            tk.StringVar(value=(MODEL_NAMES[1] if len(MODEL_NAMES) > 1 else (MODEL_NAMES[0] if MODEL_NAMES else "mistral:7b"))),
            tk.StringVar(value=(MODEL_NAMES[2] if len(MODEL_NAMES) > 2 else (MODEL_NAMES[0] if MODEL_NAMES else "qwen3:4b")))
        ]
        self.auto_model_combos = []
        for i, var in enumerate(self.auto_model_vars):
            cb = ttk.Combobox(self.model_auto_frame, textvariable=var, values=MODEL_NAMES, state="readonly")
            cb.pack(fill=tk.X, pady=(0, 5))
            self.auto_model_combos.append(cb)
        # Default: hide auto models unless mode is auto
        self.model_auto_frame.pack_forget()

        # Bind mode change
        self.mode_combo.bind("<<ComboboxSelected>>", lambda e: self.update_mode_controls())
        self.update_mode_controls()

        # Patient description
        ttk.Label(self.control_frame, text="Patient Description:").pack(anchor=tk.W)
        self.patient_desc = scrolledtext.ScrolledText(self.control_frame, wrap=tk.WORD, width=40, height=8)
        self.patient_desc.pack(fill=tk.X, pady=(0, 5))

        # Intervention
        ttk.Label(self.control_frame, text="Intervention:").pack(anchor=tk.W)
        self.intervention_entry = ttk.Entry(self.control_frame)
        self.intervention_entry.pack(fill=tk.X, pady=(0, 10))

        # Run button
        self.run_button = ttk.Button(self.control_frame, text="Run Prediction", command=self.run_selected_mode)
        self.run_button.pack(fill=tk.X, pady=(0, 10))

        # Auto-next button
        self.next_auto_button = ttk.Button(self.control_frame, text="Next Auto Case", command=self.load_next_auto_case)
        self.next_auto_button.pack(fill=tk.X, pady=(0, 10))

        # Result file selection button
        self.select_file_button = ttk.Button(self.control_frame, text="Select Result File", command=self.select_result_file)
        self.select_file_button.pack(fill=tk.X, pady=(0, 10))

        # Quit button
        self.quit_button = ttk.Button(self.control_frame, text="Quit", command=root.quit)
        self.quit_button.pack(fill=tk.X)

        # Tabs
        self.tab_control = ttk.Notebook(self.display_frame)

        self.log_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.log_tab, text="Logs")

        self.result_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.result_tab, text="Results")

        self.chart_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.chart_tab, text="Charts")

        self.tab_control.pack(fill=tk.BOTH, expand=True)

        # Navigation shortcuts
        ttk.Label(self.control_frame, text="Navigation").pack(anchor=tk.W, pady=(10, 5))
        ttk.Button(self.control_frame, text="Go to Logs", command=lambda: self.tab_control.select(self.log_tab)).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(self.control_frame, text="Go to Results", command=lambda: self.tab_control.select(self.result_tab)).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(self.control_frame, text="Go to Charts", command=lambda: self.tab_control.select(self.chart_tab)).pack(fill=tk.X, pady=(0, 5))

        # Log and result text areas
        self.log_text = scrolledtext.ScrolledText(self.log_tab, wrap=tk.WORD, width=80, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.result_text = scrolledtext.ScrolledText(self.result_tab, wrap=tk.WORD, width=80, height=20)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Chart area
        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.chart_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Stdout redirection
        self.original_stdout = None
        self.current_result_file = None

        # Auto case state
        self.auto_cases = []
        self.auto_index = 0
        self.auto_data_file = "./icu_stays_descriptions88.json"

    def redirect_stdout(self):
        """Redirect stdout to the log text widget"""
        class RedirectText:
            def __init__(self, text_widget):
                self.text_widget = text_widget
            def write(self, string):
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)
            def flush(self):
                pass
        self.original_stdout = sys.stdout
        sys.stdout = RedirectText(self.log_text)

    def restore_stdout(self):
        """Restore stdout"""
        if self.original_stdout:
            sys.stdout = self.original_stdout

    def load_next_auto_case(self):
        """Load next auto case into the input fields"""
        try:
            if not self.auto_cases:
                with open(self.auto_data_file, 'r', encoding='utf-8') as f:
                    self.auto_cases = json.load(f)
                self.auto_index = 0
                print(f"Auto cases loaded: {len(self.auto_cases)} from {self.auto_data_file}")
            if self.auto_index >= len(self.auto_cases):
                print("No more auto cases available.")
                return False
            case = self.auto_cases[self.auto_index]
            self.auto_index += 1
            desc = case.get("input_description", "")
            self.patient_desc.delete(1.0, tk.END)
            self.patient_desc.insert(tk.END, desc)
            pid = extract_patient_id(desc)
            print(f"Loaded auto case {self.auto_index}/{len(self.auto_cases)} (ID: {pid})")
            return True
        except FileNotFoundError:
            print(f"Auto data file not found: {self.auto_data_file}")
            return False
        except Exception as e:
            print(f"Error loading auto case: {str(e)}")
            return False

    def update_mode_controls(self):
        mode = self.mode_var.get()
        if mode == "auto":
            # show auto model selectors, hide single selector
            try:
                self.model_single_frame.pack_forget()
            except Exception:
                pass
            self.model_auto_frame.pack(fill=tk.X)
        else:
            # show single selector, hide auto selectors
            try:
                self.model_auto_frame.pack_forget()
            except Exception:
                pass
            self.model_single_frame.pack(fill=tk.X)

    def run_selected_mode(self):
        """Run single prediction, auto mode, or trust evaluation"""
        # Clear logs and results
        self.log_text.delete(1.0, tk.END)
        self.result_text.delete(1.0, tk.END)

        mode = self.mode_var.get()
        model_name = self.model_var.get()
        patient_info = self.patient_desc.get(1.0, tk.END).strip()
        intervention = self.intervention_entry.get().strip()

        if mode == "single" and not patient_info:
            self.log_text.insert(tk.END, "Please enter patient description before running\n")
            return

        self.run_button.config(state=tk.DISABLED)

        def run_mode_thread():
            try:
                self.redirect_stdout()
                if mode == "single":
                    if not intervention:
                        print("Error: intervention is required")
                        return
                    patient_data = {
                        "input_description": patient_info,
                        "intervention": intervention,
                        "vital_signs": {}
                    }
                    print(f"Running single-case prediction with model {model_name}")
                    prediction_file = run_prediction(model_name, patient_data)
                    if prediction_file:
                        confidence = run_evaluation(model_name, prediction_file)
                        print(f"Evaluation confidence: {confidence}")
                        try:
                            with open(prediction_file, 'r', encoding='utf-8') as f:
                                prediction_data = json.load(f)
                            best_result = {
                                "prediction_model": model_name,
                                "evaluation_models": [model_name],
                                "total_confidence": confidence if confidence is not None else 0,
                                "prediction_file": prediction_file,
                                "prediction_data": prediction_data
                            }
                            pid = extract_patient_id(patient_info, patient_data)
                            out_file = f"./result_{pid}.json"
                            save_best_prediction_result(best_result, out_file)
                            self.load_result_file(out_file)
                        except Exception as e:
                            print(f"Error generating display result file: {str(e)}")
                    else:
                        print("Prediction failed, no result file generated")
                elif mode == "auto":
                    loaded = self.load_next_auto_case()
                    auto_patient = self.patient_desc.get(1.0, tk.END).strip()
                    auto_intervention = self.intervention_entry.get().strip()
                    if not loaded or not auto_patient:
                        print("Auto mode: no case available")
                    else:
                        if not auto_intervention:
                            case_data = None
                            try:
                                if self.auto_cases and self.auto_index > 0:
                                    case_data = self.auto_cases[self.auto_index - 1]
                            except Exception:
                                case_data = None
                            expert = _lookup_expert_intervention(auto_patient, case_data)
                            if expert:
                                auto_intervention = expert
                                try:
                                    self.intervention_entry.delete(0, tk.END)
                                    self.intervention_entry.insert(0, expert)
                                except Exception:
                                    pass
                                print("Auto mode: using expert intervention from extracted file")
                            else:
                                print("Auto mode error: intervention is required")
                                return
                        # 若当前自动病例已加载，尝试从 auto_cases 中获取 output_summary 以提升比较与评分质量
                        output_summary = None
                        try:
                            if self.auto_cases and self.auto_index > 0:
                                case = self.auto_cases[self.auto_index - 1]
                                output_summary = case.get("output_summary")
                        except Exception:
                            output_summary = None
                        patient_data = {
                            "input_description": auto_patient,
                            "intervention": auto_intervention,
                            "vital_signs": {},
                            "output_summary": output_summary
                        }
                        # collect three selected models
                        selected_models = [var.get() for var in getattr(self, 'auto_model_vars', [])]
                        selected_models = [m for m in selected_models if m]
                        if len(selected_models) < 3:
                            # fallback to defaults
                            base = MODEL_NAMES[:3] if len(MODEL_NAMES) >= 3 else MODEL_NAMES
                            while len(selected_models) < min(3, len(base)):
                                selected_models.append(base[len(selected_models) % len(base)])
                        print(f"Auto rotation with models: {selected_models}")
                        # Use rotation function to get best result
                        best_result = select_best_prediction(selected_models[:3], patient_data)
                        if best_result:
                            pid = extract_patient_id(auto_patient, patient_data)
                            out_file = f"./output/best_result/result_{pid}.json"
                            save_best_prediction_result(best_result, out_file)
                            self.load_result_file(out_file)
                        else:
                            print("Auto rotation failed: no result")
                elif mode == "confidence":
                    if process_all_fact_predictions:
                        print("Starting factual prediction and trust evaluation...")
                        process_all_fact_predictions()
                        print("Completed factual prediction and trust evaluation")
                    else:
                        print("Factual prediction module unavailable, skipped")
            except Exception as e:
                print("Error running mode: {}".format(str(e)))
            finally:
                self.restore_stdout()
                self.run_button.config(state=tk.NORMAL)

        thread = threading.Thread(target=run_mode_thread)
        thread.daemon = True
        thread.start()

    def run_simple_interactive_mode(self, model_names):
        """Deprecated"""
        pass

    def select_result_file(self):
        """Select a result file and display"""
        file_path = filedialog.askopenfilename(
            title="Select Result File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            initialdir="."
        )
        if file_path:
            self.current_result_file = file_path
            self.load_result_file(file_path)

    def load_latest_result(self):
        """Load latest result file from current directory"""
        try:
            result_files = []
            for file_name in os.listdir("."):
                if file_name.startswith("result_") and file_name.endswith(".json"):
                    result_files.append((os.path.getmtime(file_name), file_name))
            if result_files:
                result_files.sort(reverse=True)
                latest_file = result_files[0][1]
                self.current_result_file = latest_file
                self.load_result_file(latest_file)
        except Exception as e:
            print(f"Error loading latest result file: {str(e)}")

    def load_result_file(self, file_path):
        """Load and display a result file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)

            # Compute SOFA scores from sofa_related_features if available
            if isinstance(result_data, dict):
                try:
                    features = None
                    if "prediction_data" in result_data:
                        features = (
                            result_data.get("prediction_data", {})
                                       .get("prediction", {})
                                       .get("intervention_analysis", {})
                                       .get("sofa_related_features")
                        )
                    elif "prediction" in result_data:
                        features = (
                            result_data.get("prediction", {})
                                       .get("intervention_analysis", {})
                                       .get("sofa_related_features")
                        )
                    if features:
                        result_data["sofa_related_features"] = features
                        def _lv(name):
                            try:
                                vals = features.get(name, [])
                                if isinstance(vals, list) and len(vals) > 0:
                                    return float(vals[-1])
                            except Exception:
                                return None
                            return None
                        computed = {
                            "sofa_respiration": calculate_sofa_respiration(_lv("pao2_fio2_ratio"), _lv("mechanical_ventilation") or 0),
                            "sofa_coagulation": calculate_sofa_coagulation(_lv("platelet")),
                            "sofa_liver": calculate_sofa_liver(_lv("bilirubin_total")),
                            "sofa_cardiovascular": calculate_sofa_cardiovascular(None, _lv("vasopressor_rate")),
                            "sofa_cns": calculate_sofa_cns(_lv("gcs_total")),
                            "sofa_renal": calculate_sofa_renal(_lv("creatinine"), _lv("urine_output_ml"))
                        }
                        computed["sofa_total"] = calculate_total_sofa(
                            computed["sofa_respiration"], computed["sofa_coagulation"], computed["sofa_liver"],
                            computed["sofa_cardiovascular"], computed["sofa_cns"], computed["sofa_renal"]
                        )
                        result_data["computed_sofa_scores"] = computed
                except Exception:
                    pass

            # Show result text
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, json.dumps(result_data, ensure_ascii=False, indent=2))

            # Plot charts
            self.plot_results(result_data)

            print(f"Loaded result file: {file_path}")
        except Exception as e:
            print(f"Error loading result file: {str(e)}")

    def plot_results(self, result_data):
        """Plot result charts using modern visualization module"""
        self.figure.clear()

        # If new visualization module is not available, fallback to original matplotlib
        if not VIZ_AVAILABLE:
            self._plot_results_fallback(result_data)
            return

        # Extract data
        sofa_scores = None
        if isinstance(result_data, dict):
            if "computed_sofa_scores" in result_data:
                sofa_scores = result_data["computed_sofa_scores"]
            elif "predicted_sofa_scores" in result_data:
                sofa_scores = result_data["predicted_sofa_scores"]
            elif "prediction_data" in result_data:
                sofa_scores = result_data.get("prediction_data", {}).get("prediction", {}).get("sofa_scores", {})

        hourly_totals = None
        if isinstance(result_data, dict) and "hourly_sofa_totals" in result_data:
            hourly_totals = result_data["hourly_sofa_totals"]

        confidence = None
        if isinstance(result_data, dict) and "total_confidence" in result_data:
            try:
                confidence = float(result_data["total_confidence"]) if result_data["total_confidence"] is not None else 0.0
            except Exception:
                confidence = 0.0

        # Determine which charts to show (preserve original logic)
        show_trend = hourly_totals is not None
        show_components = sofa_scores is not None and any(
            key.startswith("sofa_") and key != "sofa_total" for key in sofa_scores.keys()
        )
        show_confidence = confidence is not None

        # Original logic: if confidence exists, show only confidence chart
        if show_confidence:
            # Create confidence chart using matplotlib backend
            builder = ChartBuilder(backend="matplotlib")
            try:
                confidence_chart = builder.create_confidence_chart(
                    confidence=confidence,
                    title=f"模型置信度: {confidence:.2f}"
                )
                # Get the figure and copy its axes to self.figure
                ax = self.figure.add_subplot(111)
                for child_ax in confidence_chart.get_axes():
                    # Copy content from child_ax to ax
                    self._copy_axes_content(child_ax, ax)
                self.figure.tight_layout()
                self.canvas.draw()
                return
            except Exception as e:
                print(f"Error creating confidence chart: {e}")
                # Fallback to original
                self._plot_results_fallback(result_data)
                return

        # Otherwise, show trend and/or components
        num_charts = sum([show_trend, show_components])
        if num_charts == 0:
            # No data to plot
            self.canvas.draw()
            return

        current_subplot = 1
        builder = ChartBuilder(backend="matplotlib")

        # Trend chart
        if show_trend:
            try:
                trend_chart = builder.create_sofa_trend_chart(
                    hourly_totals=hourly_totals,
                    title="SOFA总分随时间变化"
                )
                ax = self.figure.add_subplot(num_charts, 1, current_subplot)
                for child_ax in trend_chart.get_axes():
                    self._copy_axes_content(child_ax, ax)
                current_subplot += 1
            except Exception as e:
                print(f"Error creating trend chart: {e}")

        # Components chart
        if show_components:
            try:
                components_chart = builder.create_sofa_component_chart(
                    sofa_scores=sofa_scores,
                    title="SOFA各组件得分"
                )
                ax = self.figure.add_subplot(num_charts, 1, current_subplot)
                for child_ax in components_chart.get_axes():
                    self._copy_axes_content(child_ax, ax)
                current_subplot += 1
            except Exception as e:
                print(f"Error creating components chart: {e}")

        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_results_fallback(self, result_data):
        """Fallback to original matplotlib plotting"""
        # SOFA scores preference: computed -> predicted -> raw prediction data
        sofa_scores = None
        if isinstance(result_data, dict):
            if "computed_sofa_scores" in result_data:
                sofa_scores = result_data["computed_sofa_scores"]
            elif "predicted_sofa_scores" in result_data:
                sofa_scores = result_data["predicted_sofa_scores"]
            elif "prediction_data" in result_data:
                sofa_scores = result_data.get("prediction_data", {}).get("prediction", {}).get("sofa_scores", {})

        # Hourly totals: time series
        if isinstance(result_data, dict) and "hourly_sofa_totals" in result_data:
            hourly_totals = result_data["hourly_sofa_totals"]
            hours = list(hourly_totals.keys())
            totals = list(hourly_totals.values())
            ax = self.figure.add_subplot(211)
            ax.plot(hours, totals, 'o-', label='SOFA total')
            ax.set_title('SOFA Total Over Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('SOFA Score')
            ax.legend()
            ax.grid(True)

        # SOFA components: bar chart
        if isinstance(sofa_scores, dict) and sofa_scores:
            systems = []
            scores = []
            for key, value in sofa_scores.items():
                if key.startswith("sofa_") and key != "sofa_total":
                    system_name = key.replace("sofa_", "").replace("_", " ")
                    systems.append(system_name)
                    try:
                        scores.append(float(value))
                    except Exception:
                        try:
                            scores.append(float(str(value)))
                        except Exception:
                            scores.append(0.0)
            ax2 = self.figure.add_subplot(212)
            ax2.bar(systems, scores)
            ax2.set_title("SOFA Component Scores")
            ax2.set_xlabel("System")
            ax2.set_ylabel("Score")
            plt.xticks(rotation=45, ha='right')

        # Confidence: pie chart
        if isinstance(result_data, dict) and "total_confidence" in result_data:
            try:
                confidence = float(result_data["total_confidence"]) if result_data["total_confidence"] is not None else 0.0
            except Exception:
                confidence = 0.0
            ax = self.figure.add_subplot(111)
            ax.pie([confidence, max(0, 100 - confidence)], labels=['Confidence', ''], autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Model Confidence: {confidence:.2f}')

        self.figure.tight_layout()
        self.canvas.draw()

    def _copy_axes_content(self, src_ax, dst_ax):
        """Copy content from source axes to destination axes"""
        # Clear destination axes
        dst_ax.clear()

        # Copy all lines
        for line in src_ax.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            label = line.get_label()
            color = line.get_color()
            linestyle = line.get_linestyle()
            linewidth = line.get_linewidth()
            marker = line.get_marker()
            markersize = line.get_markersize()

            dst_ax.plot(xdata, ydata, label=label, color=color,
                       linestyle=linestyle, linewidth=linewidth,
                       marker=marker, markersize=markersize)

        # Copy all patches (bars)
        for patch in src_ax.patches:
            try:
                # Get patch properties
                height = patch.get_height()
                width = patch.get_width()
                x = patch.get_x()
                color = patch.get_facecolor()
                # Note: patches don't have labels
                dst_ax.bar(x, height, width, color=color)
            except Exception:
                continue

        # Copy title
        if src_ax.get_title():
            dst_ax.set_title(src_ax.get_title())

        # Copy labels
        if src_ax.get_xlabel():
            dst_ax.set_xlabel(src_ax.get_xlabel())
        if src_ax.get_ylabel():
            dst_ax.set_ylabel(src_ax.get_ylabel())

        # Copy legend if present
        if src_ax.get_legend():
            dst_ax.legend()

        # Copy grid state
        try:
            # Modern matplotlib
            dst_ax.grid(src_ax.grid)
        except:
            try:
                # Older matplotlib
                dst_ax.grid(src_ax.gridOn)
            except:
                dst_ax.grid(True)

        # Copy axis limits
        dst_ax.set_xlim(src_ax.get_xlim())
        dst_ax.set_ylim(src_ax.get_ylim())

        # Copy tick parameters
        dst_ax.tick_params(axis='both', which='major', labelsize=src_ax.xaxis.get_major_ticks()[0].label.get_fontsize() if src_ax.xaxis.get_major_ticks() else 10)


def start_gui():
    """启动GUI界面"""
    try:
        import _tkinter
        root = tk.Tk()
        app = PredictionApp(root)
        root.mainloop()
        return True
    except _tkinter.TclError:
        print('图形环境不可用，无法启动GUI')
        try:
            disp = os.environ.get('DISPLAY', '')
            xdg = os.environ.get('XDG_SESSION_TYPE', '')
            print(f"诊断：DISPLAY='{disp}', XDG_SESSION_TYPE='{xdg}'")
            print("提示：若需运行 GUI，请在桌面环境或启用 X11/Wayland 图形会话。")
            print("可运行 'python3 debug_gui_env.py' 获取详细诊断与修复建议。")
        except Exception:
            pass
        raise Exception("图形环境不可用")
    except Exception as e:
        print(f'GUI 启动失败：{e}')
        raise Exception(f"GUI failed to start: {str(e)}")


if __name__ == "__main__":
    print("测试GUI模块...")
    try:
        start_gui()
        print("GUI测试成功")
    except Exception as e:
        print(f"GUI测试失败: {e}")
        sys.exit(1)
