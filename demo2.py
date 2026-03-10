import streamlit as st
import dspy
from typing import List, Dict, Union, Any
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
st.set_page_config(page_title="脓毒症智能分析助手", layout="wide")

@st.cache_resource
def configure_dspy():
    # 这里指定远程 Ollama 地址 + 模型名 qwq
    lm = dspy.LM(
        model='ollama/qwq',
        model_type='chat',
        base_url='http://172.16.51.15:11434',  # 关键点：指向远程地址
        api_key='no-key-needed'  # Ollama 不需要 API Key
    )
    
    dspy.configure(lm=lm)
    return lm

# 初始化 DSPy 配置
configure_dspy()

# --- Step 1: Define DSPy Signatures ---
class SepsisShockRiskAssessment(dspy.Signature):
    """根据患者文档评估未来12小时内进展为感染性休克的风险等级，并列出3个关键临床指标"""
    patient_document: str = dspy.InputField(desc="患者的完整临床文档")
    reasoning: str = dspy.OutputField(desc="Chain-of-thought 推理过程")
    risk_level: str = dspy.OutputField(desc="风险等级：high/medium/low")
    key_clinical_indicators: List[str] = dspy.OutputField(desc="影响判断的最主要三个临床指标")
    current_sepsis_state_summary: str = dspy.OutputField(desc="当前脓毒症状态总结，供后续模块使用")

class AssessConfidence(dspy.Signature):
    """评估输出的置信度"""
    context: str = dspy.InputField()
    output_content: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="评估推理过程")
    confidence_score: float = dspy.OutputField(desc="0-100的置信度分数")

class AnalyzeInterventionAndRisk(dspy.Signature):
    """分析干预措施的影响和风险，   """
    current_sepsis_state_summary: str = dspy.InputField()
    intervention_and_risk: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="Chain-of-thought reasoning")
    predicted_outcome: str = dspy.OutputField()
    potential_risks: List[str] = dspy.OutputField()
    risk_level: str = dspy.OutputField()
    vital_signs_trends: Dict[str, List[float]] = dspy.OutputField(desc="预测未来24小时内的生命体征趋势，每4小时一个数据点，共6个点")

class GenerateClinicalReport(dspy.Signature):
    """生成临床报告"""
    current_sepsis_state_summary: str = dspy.InputField()
    intervention_and_risk: str = dspy.InputField()
    predicted_outcome: str = dspy.InputField()
    risk_level: str = dspy.InputField()
    potential_risks: List[str] = dspy.InputField()
    vital_signs_trends: Dict[str, List[float]] = dspy.InputField(desc="预测的生命体征趋势")
    reasoning: str = dspy.OutputField(desc="Chain-of-thought reasoning")
    clinical_report: str = dspy.OutputField()

# --- Step 2: Build the PSCPA Agent Program ---
class PSCPAAgent(dspy.Program):
    def __init__(self):
        super().__init__()
        self.shock_risk_assessment = dspy.ChainOfThought(SepsisShockRiskAssessment)
        self.assess_confidence = dspy.ChainOfThought(AssessConfidence)
        self.analyze_intervention_and_risk = dspy.ChainOfThought(AnalyzeInterventionAndRisk)
        self.generate_clinical_report = dspy.ChainOfThought(GenerateClinicalReport)

    def _assess_output(self, context: str, output_content: str, step_name: str, progress_placeholder) -> tuple:
        """评估输出的置信度，如果置信度低则重试"""
        max_attempts = 3
        attempt = 1
        
        while attempt <= max_attempts:
            confidence_assessment = self.assess_confidence(
                context=context,
                output_content=output_content
            )
            
            confidence_score = confidence_assessment.confidence_score
            progress_placeholder.markdown(f"**第{attempt}次尝试** - {step_name}的置信度评分: {confidence_score:.1f}%")
            
            if confidence_score >= 80:
                return True, confidence_assessment
            
            progress_placeholder.markdown(f"🔄 置信度低于80%，正在进行第{attempt + 1}次尝试...")
            attempt += 1
            time.sleep(1)  # 添加短暂延迟以便于观察
            
        return False, confidence_assessment

    def forward(self, patient_sepsis_summary, intervention_and_risk):
        progress_placeholder = st.empty()
        assessment_history = []
        
        # Step 1: 脓毒症休克风险评估（带重试机制）
        while True:
            risk_out = self.shock_risk_assessment(
                patient_document=patient_sepsis_summary
            )
            is_confident, confidence_assessment = self._assess_output(
                context=patient_sepsis_summary,
                output_content=f"风险等级: {risk_out.risk_level}\n"
                            f"关键指标: {', '.join(risk_out.key_clinical_indicators)}\n"
                            f"当前状态总结: {risk_out.current_sepsis_state_summary}\n"
                            f"推理过程: {risk_out.reasoning}",
                step_name="感染性休克风险评估",
                progress_placeholder=progress_placeholder
            )
            assessment_history.append({
                "step": "感染性休克风险评估",
                "attempt_result": risk_out,
                "confidence_assessment": confidence_assessment
            })
            if is_confident:
                break

        # Step 2: 分析干预措施和风险（带重试机制）
        while True:
            analysis_out = self.analyze_intervention_and_risk(
                current_sepsis_state_summary=risk_out.current_sepsis_state_summary,
                intervention_and_risk=intervention_and_risk
            )
            
            # 格式化生命体征趋势以便于评估
            vital_signs_str = "\n".join([f"{k}: {v}" for k, v in analysis_out.vital_signs_trends.items()])
            
            is_confident, confidence_assessment = self._assess_output(
                context=f"状态: {risk_out.current_sepsis_state_summary}\n干预: {intervention_and_risk}",
                output_content=f"预测结果: {analysis_out.predicted_outcome}\n"
                              f"风险等级: {analysis_out.risk_level}\n"
                              f"潜在风险: {', '.join(analysis_out.potential_risks)}\n"
                              f"生命体征趋势:\n{vital_signs_str}",
                step_name="分析干预措施和风险",
                progress_placeholder=progress_placeholder
            )
            
            assessment_history.append({
                "step": "分析干预措施和风险",
                "attempt_result": analysis_out,
                "confidence_assessment": confidence_assessment
            })
            
            if is_confident:
                break

        # Step 3: 生成最终报告（无需置信度评估）
        report_out = self.generate_clinical_report(
            current_sepsis_state_summary=risk_out.current_sepsis_state_summary,
            intervention_and_risk=intervention_and_risk,
            predicted_outcome=analysis_out.predicted_outcome,
            risk_level=analysis_out.risk_level,
            potential_risks=analysis_out.potential_risks,
            vital_signs_trends=analysis_out.vital_signs_trends
        )

        # 清除进度占位符
        progress_placeholder.empty()

        return dspy.Prediction(
            reasoning_steps={
                "感染性休克风险评估": {
                    "input": {
                        "patient_sepsis_summary": patient_sepsis_summary
                    },
                    "output": {
                        "reasoning": risk_out.reasoning,
                        "risk_level": risk_out.risk_level,
                        "key_clinical_indicators": risk_out.key_clinical_indicators,
                        "current_sepsis_state_summary": risk_out.current_sepsis_state_summary,
                        "confidence_score": assessment_history[0]["confidence_assessment"].confidence_score,
                        "confidence_reasoning": assessment_history[0]["confidence_assessment"].reasoning
                    }
                },
                "分析干预措施和风险": {
                    "input": {
                        "current_sepsis_state_summary": risk_out.current_sepsis_state_summary,
                        "intervention_and_risk": intervention_and_risk
                    },
                    "output": {
                        "reasoning": analysis_out.reasoning,
                        "predicted_outcome": analysis_out.predicted_outcome,
                        "potential_risks": analysis_out.potential_risks,
                        "risk_level": analysis_out.risk_level,
                        "vital_signs_trends": analysis_out.vital_signs_trends,
                        "confidence_score": assessment_history[-1]["confidence_assessment"].confidence_score,
                        "confidence_reasoning": assessment_history[-1]["confidence_assessment"].reasoning
                    }
                },
                "生成临床报告": {
                    "input": {
                        "current_sepsis_state_summary": risk_out.current_sepsis_state_summary,
                        "intervention_and_risk": intervention_and_risk,
                        "predicted_outcome": analysis_out.predicted_outcome,
                        "risk_level": analysis_out.risk_level,
                        "potential_risks": analysis_out.potential_risks,
                        "vital_signs_trends": analysis_out.vital_signs_trends
                    },
                    "output": {
                        "reasoning": report_out.reasoning,
                        "clinical_report": report_out.clinical_report
                    }
                }
            },
            assessment_history=assessment_history,
            clinical_report=report_out.clinical_report
        )

# --- Step 4: Streamlit UI ---
st.title("💉 脓毒症智能分析助手 —— 基于DSPy与LLM")

with st.form("patient_input_form"):
    st.subheader("请填写患者信息")
    patient_info = st.text_area("患者摘要", height=400)

    intervention_and_risk = st.text_area("拟采取的干预措施和需要评估的风险", height=100)

    submit_button = st.form_submit_button("开始分析")

if submit_button:
    with st.spinner("正在调用模型进行分析..."):
        agent = PSCPAAgent()
        prediction = agent(
            patient_sepsis_summary=patient_info,
            intervention_and_risk=intervention_and_risk
        )

    st.success("✅ 分析完成！以下是完整推理过程：")

    for step_name, step_data in prediction.reasoning_steps.items():
        with st.expander(f"🔍 {step_name} - 推理过程"):
            st.markdown("### 📥 输入信息")
            for k, v in step_data["input"].items():
                key_map = {
                    "patient_sepsis_summary": "患者脓毒症摘要",
                    "current_sepsis_state_summary": "当前脓毒症状态总结",
                    "intervention_and_risk": "干预措施和风险评估",
                    "predicted_outcome": "预测结果",
                    "risk_level": "风险等级",
                    "potential_risks": "潜在风险",
                    "vital_signs_trends": "生命体征趋势"
                }
                display_key = key_map.get(k, k)
                
                if k == "vital_signs_trends" and isinstance(v, dict):
                    st.markdown(f"**{display_key}**:")
                    for sign, trend in v.items():
                        st.write(f"- {sign}: {trend}")
                else:
                    st.code(f"{display_key}: {v[:300]}...", language="text")

            st.markdown("### 📤 输出结果")
            for k, v in step_data["output"].items():
                key_map = {
                    "reasoning": "推理过程",
                    "current_sepsis_state_summary": "当前脓毒症状态总结",
                    "key_clinical_indicators": "关键临床指标",
                    "predicted_outcome": "预测结果",
                    "potential_risks": "潜在风险",
                    "risk_level": "风险等级",
                    "vital_signs_trends": "生命体征趋势预测",
                    "clinical_report": "临床报告",
                    "confidence_score": "置信度评分",
                    "confidence_reasoning": "置信度评估理由"
                }
                display_key = key_map.get(k, k)

                if isinstance(v, list):
                    st.markdown(f"**{display_key}**:")
                    for item in v:
                        st.write(f"- {item}")
                elif k == "vital_signs_trends" and isinstance(v, dict):
                    st.markdown(f"**{display_key}**:")
                    # 先显示所有数字文本
                    for sign, trend in v.items():
                        st.write(f"- {sign}: {trend}")
                    
                    # 然后显示所有趋势图 - 使用现代化Plotly图表
                    cols = st.columns(len(v))  # 创建等数量的列
                    for i, (sign, trend) in enumerate(v.items()):
                        with cols[i]:
                            if isinstance(trend, list) and len(trend) > 0:
                                # 创建Plotly图表
                                time_points = [f"T+{i*4}h" for i in range(len(trend))]

                                # 医疗主题颜色
                                medical_theme = {
                                    "primary": "#1E88E5",
                                    "secondary": "#43A047",
                                    "accent": "#FF7043",
                                    "neutral": "#78909C",
                                    "grid_color": "#E0E0E0",
                                    "background_color": "#FFFFFF",
                                    "text_color": "#212121"
                                }

                                # 生命体征颜色映射
                                vital_signs_colors = {
                                    "heart_rate": "#EF5350",      # 心率 - 红色
                                    "blood_pressure": "#AB47BC",  # 血压 - 紫色
                                    "respiratory_rate": "#29B6F6", # 呼吸频率 - 蓝色
                                    "temperature": "#FFA726",     # 体温 - 橙色
                                    "oxygen_saturation": "#66BB6A" # 血氧饱和度 - 绿色
                                }

                                # 获取颜色
                                color = vital_signs_colors.get(sign.lower(), medical_theme["primary"])

                                # 创建图表
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=time_points,
                                    y=trend,
                                    mode='lines+markers',
                                    name=sign,
                                    line=dict(color=color, width=2),
                                    marker=dict(size=6, color=color),
                                    hovertemplate='时间: %{x}<br>数值: %{y}<extra></extra>'
                                ))

                                # 更新布局
                                fig.update_layout(
                                    title=dict(
                                        text=sign,
                                        font=dict(size=14, color=medical_theme["text_color"])
                                    ),
                                    xaxis=dict(
                                        title="时间点",
                                        gridcolor=medical_theme["grid_color"],
                                        linecolor=medical_theme["neutral"],
                                        tickfont=dict(size=10)
                                    ),
                                    yaxis=dict(
                                        title="数值",
                                        gridcolor=medical_theme["grid_color"],
                                        linecolor=medical_theme["neutral"],
                                        tickfont=dict(size=10)
                                    ),
                                    plot_bgcolor=medical_theme["background_color"],
                                    paper_bgcolor=medical_theme["background_color"],
                                    font=dict(family='Arial, Helvetica, sans-serif'),
                                    margin=dict(l=40, r=30, b=40, t=40, pad=4),
                                    height=220
                                )

                                st.plotly_chart(fig, use_container_width=True, config={
                                    'displayModeBar': False
                                })
                            else:
                                st.write("无足够数据点显示趋势图")
                elif k in ["confidence_score"]:
                    st.markdown(f"**{display_key}**: {v:.1f}%")
                else:
                    st.markdown(f"**{display_key}**: {v}")

    st.subheader("📄 最终临床报告")
    st.markdown(prediction.clinical_report)