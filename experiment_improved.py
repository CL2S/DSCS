from experiment import ExperimentAgent

# 新增学习数据结构
class LearningData:
    def __init__(self):
        self.memory = defaultdict(list)  # 临床决策记忆库
        self.performance_metrics = {
            'risk_assessment_accuracy': [],
            'vital_signs_mse': [],
            'intervention_effectiveness': []
        }

    def save_memory(self, file_path):
        with open(file_path, 'w') as f:
            json.dump({
                'memory': self.memory,
                'performance_metrics': self.performance_metrics
            }, f, indent=2)

    def load_memory(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.memory = defaultdict(list, data['memory'])
            self.performance_metrics = data['performance_metrics']

class AdaptiveExperimentAgent(ExperimentAgent):
    def __init__(self):
        super().__init__()
        self.learning_data = LearningData()
        self.adaptation_rules = {
            'high_risk_missed': self._adapt_high_risk_scenario,
            'mse_threshold': self._adapt_vital_signs_model
        }
        self.state = AgentState()

    def _update_learning(self, prediction, ground_truth):
        # 经验存储
        self.learning_data.memory['clinical_cases'].append({
            'input': prediction.inputs,
            'output': prediction.outputs,
            'timestamp': datetime.now()
        })
        
        # 性能评估
        risk_acc = self._calculate_risk_accuracy(prediction, ground_truth)
        self.learning_data.performance_metrics['risk_assessment_accuracy'].append(risk_acc)
        
        # 自适应调整
        if risk_acc < 0.7:
            self.adaptation_rules['high_risk_missed'](prediction)

    def _adapt_high_risk_scenario(self, prediction):
        # 动态修改提示工程
        new_prompt = self.shock_risk_assessment.signature.instructions + """
        特别注意：当出现以下特征时应提高风险等级判断：
        - 乳酸水平>2mmol/L
        - 尿量<0.5mL/kg/h
        """
        self.shock_risk_assessment = dspy.ChainOfThought(
            self.shock_risk_assessment.signature.copy(instructions=new_prompt)
        )

    def forward(self, *args, **kwargs):
        prediction = super().forward(*args, **kwargs)
        if 'ground_truth' in kwargs:
            self._update_learning(prediction, kwargs['ground_truth'])
        return prediction

# 新增状态管理
class AgentState:
    def __init__(self):
        self.learning_cycle = 0
        self.last_adapted = None
        self.knowledge_version = 1.0
        self.performance_history = {
            'risk_assessment': [],
            'vital_signs': []
        }

    def record_performance(self, metric_type, value):
        if metric_type in self.performance_history:
            self.performance_history[metric_type].append(value)

    def get_performance_trend(self, metric_type):
        return np.mean(self.performance_history[metric_type][-5:]) if self.performance_history[metric_type] else 0

    def increment_cycle(self):
        self.learning_cycle += 1
        if self.learning_cycle % 10 == 0:
            self._update_knowledge_base()

    def _update_knowledge_base(self):
        self.knowledge_version += 0.1
        print(f"知识库更新至版本 {self.knowledge_version:.1f}")