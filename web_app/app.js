/* ==========================================================================
   Sepsis Prediction System - Main Application JavaScript
   Version: 2.0
   Vue 3 Composition API with modern state management
   ========================================================================== */

// Configuration
const CONFIG = {
  API_BASE_URL: window.location.origin,
  DEFAULT_MODELS: [
    "gemma3:12b",
    "mistral:7b",
    "qwen3:4b",
    "qwen3:30b",
    "deepseek-r1:32b",
    "medllama2:latest"
  ],
  POLLING_INTERVAL: 30000, // 30 seconds
  MAX_LOG_ENTRIES: 50,
  CHART_CONFIG: {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d']
  }
};

// Utility Functions
const Utils = {
  formatTime: (date = new Date()) => date.toLocaleTimeString(),
  formatDate: (date = new Date()) => date.toLocaleDateString(),

  debounce: (func, wait) => {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  },

  sleep: (ms) => new Promise(resolve => setTimeout(resolve, ms)),

  generateId: () => Math.random().toString(36).substring(2, 11)
};

// Chart Management
const ChartManager = {
  sofaTrendChart: null,
  componentChart: null,
  riskChart: null,

  initChart(containerId, data, layout, config = CONFIG.CHART_CONFIG) {
    return Plotly.newPlot(containerId, data, layout, config);
  },

  updateChart(containerId, data, layout) {
    return Plotly.react(containerId, data, layout);
  },

  createSOFATrendChart(data, darkMode) {
    console.log('Creating SOFA trend chart with data:', {
      has_prediction_data: !!data.prediction_data,
      prediction_data_keys: data.prediction_data ? Object.keys(data.prediction_data) : [],
      has_hourly_sofa_totals_in_prediction: !!data.prediction_data?.hourly_sofa_totals,
      has_hourly_sofa_totals_in_prediction_prediction: !!data.prediction_data?.prediction?.hourly_sofa_totals,
      has_hourly_sofa_totals_root: !!data.hourly_sofa_totals,
      has_baseline_sofa_totals: !!data.baseline_sofa_totals,
      // Debug nested structure
      has_prediction_prediction: !!data.prediction_data?.prediction,
      prediction_prediction_keys: data.prediction_data?.prediction ? Object.keys(data.prediction_data.prediction) : []
    });

    // Try multiple locations for SOFA data - with nested structure support
    let hourly = {};
    let source = 'none';

    // Check root level
    if (data.hourly_sofa_totals && Object.keys(data.hourly_sofa_totals).length > 0) {
      hourly = data.hourly_sofa_totals;
      source = 'root.hourly_sofa_totals';
    }
    // Check prediction_data level
    else if (data.prediction_data?.hourly_sofa_totals && Object.keys(data.prediction_data.hourly_sofa_totals).length > 0) {
      hourly = data.prediction_data.hourly_sofa_totals;
      source = 'prediction_data.hourly_sofa_totals';
    }
    // Check nested prediction.prediction level
    else if (data.prediction_data?.prediction?.hourly_sofa_totals && Object.keys(data.prediction_data.prediction.hourly_sofa_totals).length > 0) {
      hourly = data.prediction_data.prediction.hourly_sofa_totals;
      source = 'prediction_data.prediction.hourly_sofa_totals';
    }
    // Check sofa_totals variants
    else if (data.prediction_data?.sofa_totals && Object.keys(data.prediction_data.sofa_totals).length > 0) {
      hourly = data.prediction_data.sofa_totals;
      source = 'prediction_data.sofa_totals';
    }
    else if (data.sofa_totals && Object.keys(data.sofa_totals).length > 0) {
      hourly = data.sofa_totals;
      source = 'sofa_totals';
    }
    // Check for sofa_scores_series (calculate totals from series)
    else if (data.prediction_data?.prediction?.sofa_scores_series) {
      const series = data.prediction_data.prediction.sofa_scores_series;
      console.log('Found sofa_scores_series, calculating totals:', series);

      // Calculate hourly totals from series
      const componentKeys = ['sofa_respiration', 'sofa_coagulation', 'sofa_liver', 'sofa_cardiovascular', 'sofa_cns', 'sofa_renal'];
      const seriesLengths = componentKeys
        .map(key => series[key] ? series[key].length : 0)
        .filter(len => len > 0);

      if (seriesLengths.length > 0) {
        const maxLength = Math.max(...seriesLengths);
        for (let i = 0; i < maxLength; i++) {
          let total = 0;
          for (const key of componentKeys) {
            if (series[key] && i < series[key].length) {
              total += Number(series[key][i]) || 0;
            }
          }
          hourly[i.toString()] = total;
        }
        source = 'calculated from sofa_scores_series';
      }
    }

    const baseline = data.baseline_sofa_totals || null;

    console.log(`SOFA data source: ${source}`);
    console.log('Hourly data keys:', Object.keys(hourly));
    console.log('Hourly data:', hourly);

    const hours = Object.keys(hourly)
      .map(k => parseInt(k))
      .sort((a, b) => a - b);

    const interventionValues = hours.map(k => +hourly[k] || 0);
    const baselineValues = baseline ? hours.map(k => +baseline[k] || 0) : [];

    console.log('Processed chart data:', {
      hours,
      interventionValues,
      hasBaseline: !!baseline,
      baselineValues
    });

    const traces = [];

    if (hours.length > 0) {
      traces.push({
        x: hours,
        y: interventionValues,
        mode: 'lines+markers',
        name: 'Intervention',
        line: { color: '#60a5fa', width: 3 },
        marker: { size: 8, symbol: 'circle' }
      });
    }

    if (baselineValues.length > 0) {
      traces.push({
        x: hours,
        y: baselineValues,
        mode: 'lines',
        name: 'Baseline',
        line: { color: '#34d399', width: 2, dash: 'dash' }
      });
    }

    const layout = {
      title: { text: 'SOFA Score Trend Over Time', font: { size: 16 } },
      xaxis: {
        title: { text: 'Hours', font: { size: 14 } },
        gridcolor: darkMode ? '#475569' : '#e2e8f0'
      },
      yaxis: {
        title: { text: 'SOFA Score', font: { size: 14 } },
        gridcolor: darkMode ? '#475569' : '#e2e8f0',
        range: [0, Math.max(...interventionValues, ...baselineValues, 4) * 1.1 || 4]
      },
      plot_bgcolor: 'transparent',
      paper_bgcolor: 'transparent',
      font: { color: darkMode ? '#f1f5f9' : '#0f172a' },
      legend: {
        x: 0.02,
        y: 0.98,
        bgcolor: darkMode ? 'rgba(30, 41, 59, 0.8)' : 'rgba(255, 255, 255, 0.8)',
        bordercolor: darkMode ? '#475569' : '#e2e8f0'
      },
      margin: { l: 60, r: 30, t: 60, b: 50 }
    };

    return { traces, layout };
  },

  createComponentChart(data, darkMode) {
    // Try multiple locations for sofa scores
    let sofaScores = null;
    let source = 'none';

    // Check root level
    if (data.predicted_sofa_scores) {
      sofaScores = data.predicted_sofa_scores;
      source = 'predicted_sofa_scores';
    }
    // Check prediction_data level
    else if (data.prediction_data?.predicted_sofa_scores) {
      sofaScores = data.prediction_data.predicted_sofa_scores;
      source = 'prediction_data.predicted_sofa_scores';
    }
    // Check nested prediction.prediction level
    else if (data.prediction_data?.prediction?.sofa_scores) {
      sofaScores = data.prediction_data.prediction.sofa_scores;
      source = 'prediction_data.prediction.sofa_scores';
    }
    // Check for sofa_scores_series (use last values from series)
    else if (data.prediction_data?.prediction?.sofa_scores_series) {
      const series = data.prediction_data.prediction.sofa_scores_series;
      const componentKeys = ['sofa_respiration', 'sofa_coagulation', 'sofa_liver', 'sofa_cardiovascular', 'sofa_cns', 'sofa_renal'];

      // Calculate last scores from series
      const lastScoresFromSeries = {};
      for (const key of componentKeys) {
        if (series[key] && Array.isArray(series[key]) && series[key].length > 0) {
          lastScoresFromSeries[key] = series[key][series[key].length - 1];
        } else {
          lastScoresFromSeries[key] = 0;
        }
      }

      if (Object.keys(lastScoresFromSeries).length > 0) {
        sofaScores = lastScoresFromSeries;
        source = 'calculated from sofa_scores_series';
      }
    }

    console.log(`Component chart data source: ${source}`);
    console.log('SOFA scores found:', sofaScores);

    if (!sofaScores) {
      console.warn('No SOFA scores found for component chart');
      return null;
    }

    const components = ['Respiration', 'Coagulation', 'Liver', 'Cardiovascular', 'CNS', 'Renal'];
    const componentKeys = ['sofa_respiration', 'sofa_coagulation', 'sofa_liver', 'sofa_cardiovascular', 'sofa_cns', 'sofa_renal'];

    const lastScores = componentKeys.map(key => {
      const score = sofaScores[key];
      // Handle both numeric values and arrays
      if (Array.isArray(score)) {
        return score.length > 0 ? Number(score[score.length - 1]) || 0 : 0;
      } else {
        return Number(score) || 0;
      }
    });

    const trace = {
      x: components,
      y: lastScores,
      type: 'bar',
      marker: {
        color: lastScores.map(score => {
          if (score >= 3) return '#ef4444';
          if (score >= 2) return '#f59e0b';
          return '#10b981';
        })
      },
      text: lastScores.map(score => score.toString()),
      textposition: 'auto'
    };

    const layout = {
      title: { text: 'SOFA Component Scores (Latest)', font: { size: 16 } },
      xaxis: {
        title: { text: 'Component', font: { size: 14 } },
        tickangle: -45
      },
      yaxis: {
        title: { text: 'Score', font: { size: 14 } },
        range: [0, 4]
      },
      plot_bgcolor: 'transparent',
      paper_bgcolor: 'transparent',
      font: { color: darkMode ? '#f1f5f9' : '#0f172a' }
    };

    return { traces: [trace], layout };
  },

  createRiskChart(data, darkMode) {
    // For single mode: show single model risk assessment
    // For auto mode: show per_model_risk comparison

    console.log('Creating risk chart with data:', {
      has_per_model_risk: !!data.per_model_risk,
      has_prediction_data_risk: !!data.prediction_data?.risk_level,
      has_prediction_data_weighted_risk: !!data.prediction_data?.weighted_risk,
      prediction_data_risk_level: data.prediction_data?.risk_level,
      prediction_data_weighted_risk: data.prediction_data?.weighted_risk
    });

    // Check if we have per_model_risk (auto mode)
    if (data.per_model_risk && Array.isArray(data.per_model_risk) && data.per_model_risk.length > 0) {
      console.log('Using per_model_risk data for auto mode');
      const models = data.per_model_risk.map(item => item.model);
      const scores = data.per_model_risk.map(item => item.risk_score || 0);
      const riskLevels = data.per_model_risk.map(item => item.risk_level || 'Unknown');

      const trace = {
        x: models,
        y: scores,
        type: 'bar',
        marker: {
          color: scores.map(score => {
            if (score > 0.7) return '#ef4444';
            if (score > 0.4) return '#f59e0b';
            return '#10b981';
          })
        },
        text: riskLevels,
        textposition: 'auto',
        hovertemplate: '<b>%{x}</b><br>Score: %{y:.2f}<br>Risk: %{text}<extra></extra>'
      };

      const layout = {
        title: { text: 'Model Risk Assessment', font: { size: 16 } },
        xaxis: {
          title: { text: 'Model', font: { size: 14 } },
          tickangle: -45
        },
        yaxis: {
          title: { text: 'Risk Score', font: { size: 14 } },
          range: [0, 1]
        },
        plot_bgcolor: 'transparent',
        paper_bgcolor: 'transparent',
        font: { color: darkMode ? '#f1f5f9' : '#0f172a' }
      };

      return { traces: [trace], layout };
    }
    // For single mode: create a simple risk gauge chart
    else {
      console.log('Creating single model risk chart for single mode');

      // Try to get risk level from various sources
      let riskLevel = 'Unknown';
      let riskScore = 0;

      if (data.prediction_data?.risk_level) {
        riskLevel = data.prediction_data.risk_level;
        // Map risk level to score
        const riskMap = {
          'low': 0.2,
          'moderate': 0.5,
          'high': 0.8,
          'critical': 1.0,
          'unknown': 0.3
        };
        riskScore = riskMap[riskLevel.toLowerCase()] || 0.3;
      } else if (data.prediction_data?.weighted_risk) {
        riskLevel = data.prediction_data.weighted_risk;
        // Try to parse numeric score from string
        const match = riskLevel.match(/[\d.]+/);
        if (match) {
          riskScore = parseFloat(match[0]) / 100; // Assume percentage
        } else {
          const riskMap = {
            'low': 0.2,
            'moderate': 0.5,
            'high': 0.8,
            'critical': 1.0,
            'unknown': 0.3
          };
          riskScore = riskMap[riskLevel.toLowerCase()] || 0.3;
        }
      } else if (data.final_weighted_risk_level) {
        riskLevel = data.final_weighted_risk_level;
        riskScore = 0.6; // Default moderate
      } else if (data.risk_level) {
        riskLevel = data.risk_level;
        riskScore = 0.6; // Default moderate
      }

      console.log('Single mode risk:', { riskLevel, riskScore });

      // Create gauge chart for single risk
      const trace = {
        type: "indicator",
        mode: "gauge+number",
        value: riskScore,
        title: { text: "Risk Level", font: { size: 16 } },
        gauge: {
          axis: { range: [0, 1], tickwidth: 1, tickcolor: darkMode ? "#f1f5f9" : "#0f172a" },
          bar: { color: riskScore > 0.7 ? '#ef4444' : riskScore > 0.4 ? '#f59e0b' : '#10b981', thickness: 0.3 },
          bgcolor: darkMode ? "#1e293b" : "#f8fafc",
          borderwidth: 2,
          bordercolor: darkMode ? "#475569" : "#e2e8f0",
          steps: [
            { range: [0, 0.4], color: "#10b981" },
            { range: [0.4, 0.7], color: "#f59e0b" },
            { range: [0.7, 1], color: "#ef4444" }
          ],
          threshold: {
            line: { color: "#dc2626", width: 4 },
            thickness: 0.75,
            value: 0.8
          }
        },
        number: { font: { size: 20 } }
      };

      const layout = {
        title: { text: `Risk Assessment: ${riskLevel}`, font: { size: 18 } },
        plot_bgcolor: 'transparent',
        paper_bgcolor: 'transparent',
        font: { color: darkMode ? '#f1f5f9' : '#0f172a' },
        margin: { t: 50, b: 20, l: 50, r: 50 }
      };

      return { traces: [trace], layout };
    }
  }
};

// API Client
const ApiClient = {
  async get(endpoint) {
    try {
      const response = await axios.get(`${CONFIG.API_BASE_URL}${endpoint}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  async post(endpoint, data) {
    try {
      const response = await axios.post(`${CONFIG.API_BASE_URL}${endpoint}`, data);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  handleError(error) {
    let errorMessage = 'Unknown error';

    if (error.response) {
      // Server responded with error status
      const status = error.response.status;
      const data = error.response.data;

      // Build detailed error message
      let details = '';
      if (typeof data === 'string') {
        details = data;
      } else if (data && typeof data === 'object') {
        // Try to extract useful error information
        if (data.error) details = data.error;
        else if (data.message) details = data.message;
        else if (data.detail) details = data.detail;
        else details = JSON.stringify(data, null, 2);
      }

      errorMessage = `API Error: ${status}`;
      if (details) {
        errorMessage += ` - ${details}`;
      }

      // Add specific error handling for common status codes
      switch(status) {
        case 400:
          errorMessage += ' (Bad Request: Check your input data)';
          break;
        case 401:
          errorMessage += ' (Unauthorized: Check authentication)';
          break;
        case 403:
          errorMessage += ' (Forbidden: Insufficient permissions)';
          break;
        case 404:
          errorMessage += ' (Not Found: Endpoint or resource not found)';
          break;
        case 500:
          errorMessage += ' (Internal Server Error: Backend service issue)';
          break;
        case 502:
          errorMessage += ' (Bad Gateway: Proxy or upstream service issue)';
          break;
        case 503:
          errorMessage += ' (Service Unavailable: Backend service down)';
          break;
        case 504:
          errorMessage += ' (Gateway Timeout: Request took too long)';
          break;
      }

    } else if (error.request) {
      // Request made but no response
      errorMessage = 'Network Error: Unable to reach the server. ';
      errorMessage += 'Please check: 1) Server is running, 2) CORS configuration, 3) Network connectivity';
    } else {
      // Error setting up the request
      errorMessage = `Request Error: ${error.message}`;
    }

    // Log full error for debugging
    console.error('API Error Details:', {
      error,
      response: error.response,
      request: error.request,
      message: error.message,
      config: error.config
    });

    return new Error(errorMessage);
  }
};

// Vue Application
const { createApp, ref, computed, onMounted, watch } = Vue;

const app = createApp({
  setup() {
    // Theme Management
    const darkMode = ref(true);
    const toggleTheme = () => {
      darkMode.value = !darkMode.value;
      document.documentElement.classList.toggle('dark', darkMode.value);
      appState.addLog('info', `Switched to ${darkMode.value ? 'dark' : 'light'} theme`);
    };

    // Application State
    const appState = {
      mode: ref('single'),
      selectedModel: ref(''),
      autoModels: ref(['', '', '']),
      patientInput: ref(''),
      intervention: ref(''),
      loading: ref(false),
      activeTab: ref('results'),
      lastUpdate: ref(Utils.formatTime()),

      models: ref([]),
      result: ref({
        reasoning: '',
        modelResults: [],
        rawData: null
      }),
      kpi: ref({
        model: '',
        risk: '',
        confidence: '',
        sofa: ''
      }),
      logs: ref([]),
      patients: ref([]),
      connectionStatus: ref('connecting'),
      patientsError: ref(''),
      chartData: ref({
        sofaTrend: null,
        componentChart: null,
        riskChart: null
      }),

      addLog(type, message) {
        const time = Utils.formatTime();
        appState.logs.value.unshift({ time, type, message, id: Utils.generateId() });
        if (appState.logs.value.length > CONFIG.MAX_LOG_ENTRIES) {
          appState.logs.value.pop();
        }
        appState.lastUpdate.value = time;
      },

      clearLogs() {
        appState.logs.value = [];
        appState.addLog('info', 'Logs cleared');
      },

      logClass(type) {
        switch(type) {
          case 'error': return 'log-error';
          case 'success': return 'log-success';
          case 'warning': return 'log-warning';
          default: return 'log-info';
        }
      }
    };

    // Computed Properties
    const computedProps = {
      statusText: computed(() => {
        switch(appState.connectionStatus.value) {
          case 'connected': return 'Connected';
          case 'connecting': return 'Connecting...';
          case 'error': return 'Connection Error';
          default: return 'Unknown';
        }
      }),

      statusClass: computed(() => {
        switch(appState.connectionStatus.value) {
          case 'connected': return 'status-success';
          case 'connecting': return 'status-warning';
          case 'error': return 'status-danger';
          default: return 'status-warning';
        }
      }),

      runButtonText: computed(() => {
        if (appState.loading.value) return 'Running...';
        switch(appState.mode.value) {
          case 'single': return 'Run Single Prediction';
          case 'auto': return 'Run Auto Comparison';
          case 'confidence': return 'Evaluate Confidence';
          default: return 'Run Prediction';
        }
      }),

      riskClass: computed(() => {
        const risk = appState.kpi.value.risk?.toLowerCase();
        if (risk?.includes('high')) return 'text-danger';
        if (risk?.includes('medium')) return 'text-warning';
        if (risk?.includes('low')) return 'text-success';
        return '';
      }),

      hasResults: computed(() => {
        return !!appState.result.value.rawData ||
               !!appState.result.value.reasoning ||
               appState.result.value.modelResults.length > 0;
      })
    };

    // Business Logic
    const businessLogic = {
      getRiskClass(risk) {
        if (!risk) return 'status-warning';
        const riskLower = risk.toLowerCase();
        if (riskLower.includes('high')) return 'status-danger';
        if (riskLower.includes('medium')) return 'status-warning';
        if (riskLower.includes('low')) return 'status-success';
        return 'status-warning';
      },

      extractLatestSOFA(data) {
        // Try multiple possible locations for SOFA data
        let sofaTotals = null;

        if (data.prediction_data?.hourly_sofa_totals) {
          sofaTotals = data.prediction_data.hourly_sofa_totals;
          console.log('Found SOFA totals in prediction_data.hourly_sofa_totals');
        } else if (data.hourly_sofa_totals) {
          sofaTotals = data.hourly_sofa_totals;
          console.log('Found SOFA totals in hourly_sofa_totals');
        } else if (data.sofa_totals) {
          sofaTotals = data.sofa_totals;
          console.log('Found SOFA totals in sofa_totals');
        } else if (data.prediction_data?.sofa_scores) {
          sofaTotals = data.prediction_data.sofa_scores;
          console.log('Found SOFA totals in prediction_data.sofa_scores');
        }

        if (sofaTotals && typeof sofaTotals === 'object') {
          const totals = Object.values(sofaTotals);
          const latest = totals.length > 0 ? totals[totals.length - 1] : null;
          console.log('Extracted latest SOFA score:', latest);
          return latest;
        }

        console.warn('No SOFA totals found in response');
        return null;
      },

      async initializeApp() {
        try {
          appState.addLog('info', 'Initializing application...');

          // Load models from API
          const modelsData = await ApiClient.get('/api/models');
          appState.models.value = modelsData.models || CONFIG.DEFAULT_MODELS;

          if (appState.models.value.length > 0) {
            appState.selectedModel.value = appState.models.value[0];
            appState.autoModels.value = appState.models.value.slice(0, 3);
          }

          // Load patients
          await businessLogic.loadPatients();

          appState.connectionStatus.value = 'connected';
          appState.addLog('success', 'Application initialized successfully');
        } catch (error) {
          appState.connectionStatus.value = 'error';
          appState.addLog('error', `Initialization failed: ${error.message}`);

          // Fallback to default models
          appState.models.value = CONFIG.DEFAULT_MODELS;
          appState.selectedModel.value = appState.models.value[0];
          appState.autoModels.value = appState.models.value.slice(0, 3);
          appState.addLog('info', 'Using fallback model list');
        }
      },

      async loadPatients() {
        try {
          appState.patientsError.value = '';
          appState.addLog('info', 'Loading patients...');
          console.log('Calling /api/patients/list endpoint');
          const patientsData = await ApiClient.get('/api/patients/list?limit=10');
          console.log('Patients API response:', patientsData);
          appState.patients.value = patientsData.patients || [];
          if (appState.patients.value.length === 0) {
            appState.patientsError.value = 'No patients found in database. Database may be empty or not configured.';
          }
          appState.addLog('success', `Loaded ${appState.patients.value.length} patients`);
        } catch (error) {
          const errorMsg = `Failed to load patients: ${error.message}`;
          appState.patientsError.value = errorMsg;
          appState.addLog('error', errorMsg);
          console.error('Load patients error:', error);
          appState.patients.value = [];
        }
      },

      loadPatient(patient) {
        console.log('Loading patient:', patient);
        // Try both field names for compatibility
        const description = patient.input_description || patient.description || '';
        appState.patientInput.value = description;
        appState.addLog('info', `Loaded patient ${patient.id}`);
        // Also switch to single mode and update model if available
        appState.mode.value = 'single';
      },

      async runPrediction() {
        if (appState.loading.value) return;

        appState.loading.value = true;
        appState.addLog('info', `Starting ${appState.mode.value} prediction...`);

        try {
          let response;
          const baseData = { model: appState.selectedModel.value };

          // Log the data being sent for debugging
          console.log('Sending prediction data:', {
            mode: appState.mode.value,
            baseData,
            patientInput: appState.patientInput.value,
            intervention: appState.intervention.value,
            autoModels: appState.autoModels.value
          });

          let requestData;
          switch(appState.mode.value) {
            case 'single':
              requestData = {
                ...baseData,
                input_description: appState.patientInput.value,
                intervention: appState.intervention.value
              };
              console.log('Single prediction request data:', requestData);
              response = await ApiClient.post('/api/run_single', requestData);
              break;

            case 'auto':
              requestData = {
                models: appState.autoModels.value.filter(m => m),
                input_description: appState.patientInput.value,
                intervention: appState.intervention.value
              };
              console.log('Auto prediction request data:', requestData);
              response = await ApiClient.post('/api/run_auto', requestData);
              break;

            case 'confidence':
              requestData = baseData;
              console.log('Confidence evaluation request data:', requestData);
              response = await ApiClient.post('/api/confidence', requestData);
              break;
          }

          if (response) {
            console.log('Prediction response received:', response);
            businessLogic.handlePredictionResult(response);
            appState.activeTab.value = 'results';
            appState.addLog('success', `${appState.mode.value} prediction completed successfully`);
          }
        } catch (error) {
          const errorMsg = `Prediction failed: ${error.message}`;
          appState.addLog('error', errorMsg);
          console.error('Prediction error details:', error);

          // Show detailed error in alert for debugging
          setTimeout(() => {
            alert(`Prediction Error:\n\n${error.message}\n\nCheck browser console for details.`);
          }, 100);
        } finally {
          appState.loading.value = false;
        }
      },

      handlePredictionResult(data) {
        console.log('Raw API response data:', data);
        console.log('Current mode:', appState.mode.value);

        // Store raw data
        appState.result.value.rawData = data;

        // Debug: Check data structure
        console.log('Full API response data structure:', JSON.stringify(data, null, 2));

        // Different processing based on mode
        if (appState.mode.value === 'single') {
          console.log('Processing SINGLE prediction result');
          console.log('Single prediction data structure:', {
            has_prediction_data: !!data.prediction_data,
            prediction_data_keys: data.prediction_data ? Object.keys(data.prediction_data) : [],
            prediction_model: data.prediction_model,
            evaluation_models: data.evaluation_models,
            total_confidence: data.total_confidence,
            has_baseline_sofa_totals: !!data.baseline_sofa_totals,
            baseline_sofa_totals_keys: data.baseline_sofa_totals ? Object.keys(data.baseline_sofa_totals) : [],
            has_output_summary: !!data.output_summary,
            full_data_keys: Object.keys(data)
          });

          // Debug: Check for SOFA data in prediction_data
          if (data.prediction_data) {
            console.log('Prediction data SOFA info:', {
              has_hourly_sofa_totals: !!data.prediction_data.hourly_sofa_totals,
              hourly_sofa_totals_keys: data.prediction_data.hourly_sofa_totals ? Object.keys(data.prediction_data.hourly_sofa_totals) : [],
              has_predicted_sofa_scores: !!data.prediction_data.predicted_sofa_scores,
              predicted_sofa_scores_keys: data.prediction_data.predicted_sofa_scores ? Object.keys(data.prediction_data.predicted_sofa_scores) : [],
              has_prediction_reasoning: !!data.prediction_data.prediction_reasoning,
              has_risk_level: !!data.prediction_data.risk_level,
              has_weighted_risk: !!data.prediction_data.weighted_risk,
              // Deep dive into prediction_data structure
              prediction_data_patient_id: data.prediction_data.patient_id,
              prediction_data_model_name: data.prediction_data.model_name,
              prediction_data_prediction_keys: data.prediction_data.prediction ? Object.keys(data.prediction_data.prediction) : [],
              has_prediction_intervention_analysis: !!data.prediction_data.prediction?.intervention_analysis,
              prediction_intervention_analysis_keys: data.prediction_data.prediction?.intervention_analysis ? Object.keys(data.prediction_data.prediction.intervention_analysis) : []
            });

            // Deep dive into prediction structure if exists
            if (data.prediction_data.prediction) {
              console.log('Prediction structure:', {
                intervention_analysis: data.prediction_data.prediction.intervention_analysis,
                has_sofa_scores_series: !!data.prediction_data.prediction.sofa_scores_series,
                has_hourly_sofa_totals_in_prediction: !!data.prediction_data.prediction.hourly_sofa_totals,
                has_sofa_scores_in_prediction: !!data.prediction_data.prediction.sofa_scores
              });
            }
          }

          // For single mode, extract from prediction_data
          const predictionData = data.prediction_data || {};

          // Update KPI for single mode
          const modelName = data.prediction_model || '-';
          const riskLevel = predictionData.risk_level || predictionData.weighted_risk || '-';
          const confidenceValue = data.total_confidence;
          const confidenceStr = confidenceValue !== undefined ? confidenceValue.toFixed(2) : '-';
          const sofaValue = businessLogic.extractLatestSOFA(data) || '-';

          console.log('Single mode KPI values:', { modelName, riskLevel, confidenceStr, sofaValue });

          appState.kpi.value = {
            model: modelName,
            risk: riskLevel,
            confidence: confidenceStr,
            sofa: sofaValue
          };

          // Update reasoning for single mode - comprehensive search
          let reasoningText = 'No reasoning available';

          // Debug: Log complete data structure for single mode reasoning search
          console.log('Single mode complete data structure for reasoning search:');

          // Deep dive into prediction_data structure
          if (predictionData) {
            console.log('predictionData structure:', {
              keys: Object.keys(predictionData),
              hasPrediction: !!predictionData.prediction,
              predictionKeys: predictionData.prediction ? Object.keys(predictionData.prediction) : [],
              hasInterventionAnalysis: !!predictionData.prediction?.intervention_analysis,
              interventionAnalysisKeys: predictionData.prediction?.intervention_analysis ? Object.keys(predictionData.prediction.intervention_analysis) : []
            });

            // Search recursively for reasoning fields
            const searchPaths = [
              // Direct fields in predictionData
              predictionData.prediction_reasoning,
              predictionData.reasoning,
              predictionData.explanation,

              // Nested in prediction
              predictionData.prediction?.intervention_analysis?.reasoning,
              predictionData.prediction?.stages?.reasoning,
              predictionData.prediction?.explanation,

              // Deep nested paths
              predictionData.prediction?.intervention_analysis?.prediction_reasoning,
              predictionData.prediction?.stages?.prediction_reasoning,

              // Check for any text fields in intervention_analysis
              predictionData.prediction?.intervention_analysis?.analysis,
              predictionData.prediction?.intervention_analysis?.summary,
              predictionData.prediction?.intervention_analysis?.details
            ];

            const pathNames = [
              'predictionData.prediction_reasoning',
              'predictionData.reasoning',
              'predictionData.explanation',
              'predictionData.prediction.intervention_analysis.reasoning',
              'predictionData.prediction.stages.reasoning',
              'predictionData.prediction.explanation',
              'predictionData.prediction.intervention_analysis.prediction_reasoning',
              'predictionData.prediction.stages.prediction_reasoning',
              'predictionData.prediction.intervention_analysis.analysis',
              'predictionData.prediction.intervention_analysis.summary',
              'predictionData.prediction.intervention_analysis.details'
            ];

            console.log('Searching through reasoning paths:');
            for (let i = 0; i < searchPaths.length; i++) {
              const value = searchPaths[i];
              if (value && typeof value === 'string' && value.trim().length > 0) {
                console.log(`✓ Found at ${pathNames[i]}: ${value.substring(0, 100)}...`);
                reasoningText = value;
                break;
              } else if (value) {
                console.log(`  ${pathNames[i]}: ${typeof value} (not suitable)`);
              } else {
                console.log(`  ${pathNames[i]}: not found`);
              }
            }
          }

          // If still not found, check root data fields
          if (!reasoningText || reasoningText === 'No reasoning available') {
            console.log('Checking root data fields for reasoning:');

            const rootPaths = [
              data.reasoning,
              data.prediction_reasoning,
              data.explanation,
              data.analysis,
              data.summary
            ];

            const rootNames = [
              'data.reasoning',
              'data.prediction_reasoning',
              'data.explanation',
              'data.analysis',
              'data.summary'
            ];

            for (let i = 0; i < rootPaths.length; i++) {
              const value = rootPaths[i];
              if (value && typeof value === 'string' && value.trim().length > 0) {
                console.log(`✓ Found at ${rootNames[i]}: ${value.substring(0, 100)}...`);
                reasoningText = value;
                break;
              }
            }
          }

          // Final fallback
          if (!reasoningText || reasoningText === 'No reasoning available') {
            console.warn('No reasoning text found in any location in single mode');

            // Try to extract any text from the entire response
            const jsonString = JSON.stringify(data);
            const keywords = ['reasoning', 'explanation', 'analysis', 'summary', 'prediction'];
            for (const keyword of keywords) {
              const regex = new RegExp(`"${keyword}":\\s*"([^"]+)"`, 'i');
              const match = jsonString.match(regex);
              if (match && match[1]) {
                reasoningText = match[1];
                console.log(`Extracted reasoning using keyword "${keyword}": ${reasoningText.substring(0, 100)}...`);
                break;
              }
            }
          }

          console.log('Single mode reasoning text:', reasoningText);
          appState.result.value.reasoning = reasoningText;

          // Single mode doesn't have model comparison, but we can show model info
          if (data.evaluation_models && Array.isArray(data.evaluation_models)) {
            appState.result.value.modelResults = data.evaluation_models.map(model => ({
              model: model,
              score: data.total_confidence ? data.total_confidence.toFixed(2) : 'N/A',
              risk: riskLevel || 'Unknown'
            }));
          } else {
            appState.result.value.modelResults = [];
          }

        } else if (appState.mode.value === 'auto') {
          console.log('Processing AUTO prediction result');
          console.log('Auto prediction data fields:', {
            prediction_model: data.prediction_model,
            best_model: data.best_model,
            model_name: data.model_name,
            final_weighted_risk_level: data.final_weighted_risk_level,
            risk_level: data.risk_level,
            weighted_risk: data.weighted_risk,
            total_confidence: data.total_confidence,
            confidence_score: data.confidence_score,
            has_prediction_data: !!data.prediction_data,
            prediction_reasoning: data.prediction_data?.prediction_reasoning,
            reasoning: data.reasoning,
            per_model_risk: data.per_model_risk,
            model_results: data.model_results,
            sofa_data: businessLogic.extractLatestSOFA(data)
          });

          // Update KPI with flexible field mapping for auto mode
          const modelName = data.prediction_model || data.best_model || data.model_name || '-';
          const riskLevel = data.final_weighted_risk_level || data.risk_level || data.weighted_risk || '-';
          const confidenceValue = data.total_confidence || data.confidence_score;
          const confidenceStr = confidenceValue ? confidenceValue.toFixed(2) : '-';
          const sofaValue = businessLogic.extractLatestSOFA(data) || '-';

          console.log('Auto mode KPI values:', { modelName, riskLevel, confidenceStr, sofaValue });

          appState.kpi.value = {
            model: modelName,
            risk: riskLevel,
            confidence: confidenceStr,
            sofa: sofaValue
          };

          // Update reasoning for auto mode - find reasoning from the model with highest score
          let reasoningText = 'No reasoning available';

          // Debug: Log complete data structure for reasoning search
          console.log('Auto mode complete data structure for reasoning search:');
          console.log('Full data keys:', Object.keys(data));

          if (data.per_model_risk) {
            console.log('per_model_risk array details:', {
              length: data.per_model_risk.length,
              firstItemKeys: data.per_model_risk[0] ? Object.keys(data.per_model_risk[0]) : 'empty',
              firstItemFull: data.per_model_risk[0],
              allItems: data.per_model_risk.map(item => ({
                model: item.model,
                risk_score: item.risk_score,
                risk_level: item.risk_level,
                confidence: item.confidence,
                hasReasoning: !!item.reasoning,
                reasoningLength: item.reasoning?.length || 0,
                hasExplanation: !!item.explanation,
                allKeys: Object.keys(item)
              }))
            });
          }

          if (data.model_results) {
            console.log('model_results array details:', {
              length: data.model_results.length,
              firstItemKeys: data.model_results[0] ? Object.keys(data.model_results[0]) : 'empty',
              firstItemFull: data.model_results[0],
              allItems: data.model_results.map(item => ({
                model: item.model || item.model_name,
                confidence: item.confidence,
                score: item.score,
                hasReasoning: !!item.reasoning,
                reasoningLength: item.reasoning?.length || 0,
                hasExplanation: !!item.explanation,
                allKeys: Object.keys(item)
              }))
            });
          }

          // Strategy 1: Look for reasoning in per_model_risk array (highest confidence or lowest risk)
          let bestModelReasoning = null;
          let bestModelScore = -Infinity;
          let bestModelName = '';

          // Check per_model_risk array
          if (data.per_model_risk && Array.isArray(data.per_model_risk) && data.per_model_risk.length > 0) {
            console.log('Searching for best model in per_model_risk array');

            for (const item of data.per_model_risk) {
              // Calculate score - higher is better
              // Use confidence if available, otherwise use inverse of risk_score
              let score = 0;
              if (item.confidence !== undefined) {
                score = item.confidence;
              } else if (item.risk_score !== undefined) {
                // Lower risk_score is better, so invert (1 - risk_score) gives higher score for lower risk
                score = 1 - item.risk_score;
              } else {
                score = 0;
              }

              console.log(`Model ${item.model}: score=${score}, reasoning=${item.reasoning ? 'Yes' : 'No'}`);

              if (score > bestModelScore) {
                bestModelScore = score;
                bestModelName = item.model || 'Unknown';

                // Check for reasoning or explanation
                if (item.reasoning) {
                  bestModelReasoning = item.reasoning;
                  console.log(`New best model: ${bestModelName} with score ${bestModelScore}, has reasoning`);
                } else if (item.explanation) {
                  bestModelReasoning = item.explanation;
                  console.log(`New best model: ${bestModelName} with score ${bestModelScore}, has explanation`);
                } else {
                  bestModelReasoning = null;
                  console.log(`New best model: ${bestModelName} with score ${bestModelScore}, no reasoning found`);
                }
              }
            }
          }

          // Strategy 2: Check model_results array if no reasoning found yet
          if (!bestModelReasoning && data.model_results && Array.isArray(data.model_results) && data.model_results.length > 0) {
            console.log('Searching for best model in model_results array');

            for (const item of data.model_results) {
              // Calculate score based on confidence or score field
              let score = 0;
              if (item.confidence !== undefined) {
                score = item.confidence;
              } else if (item.score !== undefined) {
                score = item.score;
              }

              console.log(`Model ${item.model || item.model_name}: score=${score}, reasoning=${item.reasoning ? 'Yes' : 'No'}`);

              if (score > bestModelScore) {
                bestModelScore = score;
                bestModelName = item.model || item.model_name || 'Unknown';

                if (item.reasoning) {
                  bestModelReasoning = item.reasoning;
                  console.log(`New best model: ${bestModelName} with score ${bestModelScore}, has reasoning`);
                } else if (item.explanation) {
                  bestModelReasoning = item.explanation;
                  console.log(`New best model: ${bestModelName} with score ${bestModelScore}, has explanation`);
                }
              }
            }
          }

          // Use best model reasoning if found
          if (bestModelReasoning) {
            reasoningText = bestModelReasoning;
            console.log(`Using reasoning from best model "${bestModelName}" with score ${bestModelScore}`);
          } else {
            // Fallback to general reasoning fields
            console.log('No reasoning found in model arrays, trying general fields');

            const generalReasoningOptions = [
              { field: data.prediction_data?.prediction_reasoning, name: 'prediction_data.prediction_reasoning' },
              { field: data.reasoning, name: 'data.reasoning' },
              { field: data.prediction_reasoning, name: 'data.prediction_reasoning' },
              { field: data.explanation, name: 'data.explanation' },
              { field: data.prediction_data?.explanation, name: 'prediction_data.explanation' },
              { field: data.prediction_data?.reasoning, name: 'prediction_data.reasoning' }
            ];

            for (const option of generalReasoningOptions) {
              if (option.field) {
                reasoningText = option.field;
                console.log(`Using reasoning from ${option.name}`);
                break;
              }
            }
          }

          console.log('Auto mode reasoning text:', reasoningText);
          appState.result.value.reasoning = reasoningText;

          // Update model results - try multiple possible field names for auto mode
          let modelResultsArray = null;

          if (data.per_model_risk && Array.isArray(data.per_model_risk)) {
            modelResultsArray = data.per_model_risk;
            console.log('Using per_model_risk array with', modelResultsArray.length, 'items');
          } else if (data.model_results && Array.isArray(data.model_results)) {
            modelResultsArray = data.model_results;
            console.log('Using model_results array with', modelResultsArray.length, 'items');
          } else if (data.models && Array.isArray(data.models)) {
            modelResultsArray = data.models;
            console.log('Using models array with', modelResultsArray.length, 'items');
          }

          if (modelResultsArray && modelResultsArray.length > 0) {
            appState.result.value.modelResults = modelResultsArray.map(item => ({
              model: item.model || item.model_name || item.name || 'Unknown',
              score: item.risk_score?.toFixed(2) || item.score?.toFixed(2) || item.confidence?.toFixed(2) || 'N/A',
              risk: item.risk_level || item.risk || item.risk_category || 'Unknown'
            }));
            console.log('Processed model results:', appState.result.value.modelResults);
          } else {
            console.warn('No model results array found in response. Available keys:', Object.keys(data).filter(k => Array.isArray(data[k])));
            appState.result.value.modelResults = [];
          }
        } else if (appState.mode.value === 'confidence') {
          console.log('Processing CONFIDENCE evaluation result');
          // Handle confidence mode separately if needed
          appState.kpi.value = {
            model: data.model || '-',
            risk: '-',
            confidence: data.confidence_score ? data.confidence_score.toFixed(2) : '-',
            sofa: '-'
          };
          appState.result.value.reasoning = data.explanation || 'Confidence evaluation completed';
          appState.result.value.modelResults = [];
        }

        // Debug: Check updated state and reasoning extraction
        console.log('Updated appState:', {
          kpi: appState.kpi.value,
          reasoning: appState.result.value.reasoning ? appState.result.value.reasoning.substring(0, 100) + '...' : 'empty',
          reasoningLength: appState.result.value.reasoning?.length || 0,
          modelResults: appState.result.value.modelResults,
          hasChartData: !!appState.chartData.value.sofaTrend,
          chartDataSources: {
            sofaTrend: appState.chartData.value.sofaTrend ? 'has data' : 'no data',
            componentChart: appState.chartData.value.componentChart ? 'has data' : 'no data',
            riskChart: appState.chartData.value.riskChart ? 'has data' : 'no data'
          }
        });

        // Additional debug: Deep search for reasoning if empty
        if (!appState.result.value.reasoning || appState.result.value.reasoning === 'No reasoning available') {
          console.warn('Reasoning is empty or default, performing deep search');
          console.log('Complete data structure for debugging:', JSON.stringify(data, null, 2));
        }

        // Update charts
        businessLogic.updateCharts(data);
      },

      updateCharts(data) {
        console.log('Generating chart data from prediction result');

        // Generate and store chart data
        const sofaTrend = ChartManager.createSOFATrendChart(data, darkMode.value);
        const componentChart = ChartManager.createComponentChart(data, darkMode.value);
        const riskChart = ChartManager.createRiskChart(data, darkMode.value);

        appState.chartData.value = {
          sofaTrend,
          componentChart,
          riskChart
        };

        console.log('Chart data stored, ready for rendering when charts tab is active');

        // If charts tab is currently active, render immediately
        if (appState.activeTab.value === 'charts') {
          businessLogic.renderCharts();
        }
      },

      renderCharts() {
        console.log('Rendering charts for active tab');
        const { sofaTrend, componentChart, riskChart } = appState.chartData.value;

        if (!sofaTrend || !componentChart || !riskChart) {
          console.warn('No chart data available to render');
          return;
        }

        // Helper function to safely render chart
        const safeRenderChart = (containerId, chartData) => {
          if (!chartData) return;
          const container = document.getElementById(containerId);
          if (!container) {
            console.warn(`Chart container ${containerId} not found, cannot render`);
            return;
          }

          // Check if chart already exists
          if (ChartManager[containerId]) {
            try {
              ChartManager.updateChart(containerId, chartData.traces, chartData.layout);
            } catch (error) {
              console.error(`Failed to update chart ${containerId}:`, error);
              // Fallback to creating new chart
              ChartManager[containerId] = ChartManager.initChart(containerId, chartData.traces, chartData.layout);
            }
          } else {
            ChartManager[containerId] = ChartManager.initChart(containerId, chartData.traces, chartData.layout);
          }
        };

        safeRenderChart('sofaTrendChart', sofaTrend);
        safeRenderChart('componentChart', componentChart);
        safeRenderChart('riskChart', riskChart);
      },

      resetSession() {
        appState.patientInput.value = '';
        appState.intervention.value = '';
        appState.result.value = { reasoning: '', modelResults: [], rawData: null };
        appState.kpi.value = { model: '', risk: '', confidence: '', sofa: '' };
        appState.patientsError.value = '';
        appState.addLog('info', 'Session reset');
      },

      loadExample() {
        appState.patientInput.value = '65-year-old male with pneumonia, fever 39°C, BP 90/60, HR 120, RR 28, SpO2 88% on room air. WBC 18,000, creatinine 2.1, lactate 3.8. Recent ICU admission for respiratory failure.';
        appState.intervention.value = 'Norepinephrine 5μg/kg/min, IV fluids 30mL/kg, mechanical ventilation with PEEP 10, broad-spectrum antibiotics.';
        appState.addLog('info', 'Loaded example patient data');
      },

      exportResults() {
        if (!appState.result.value.rawData) {
          appState.addLog('warning', 'No results to export');
          return;
        }

        const exportData = {
          timestamp: new Date().toISOString(),
          version: '2.0',
          mode: appState.mode.value,
          patientInput: appState.patientInput.value,
          intervention: appState.intervention.value,
          result: appState.result.value,
          kpi: appState.kpi.value,
          rawData: appState.result.value.rawData
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `sepsis_prediction_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        appState.addLog('success', 'Results exported successfully');
      },

      showHelp() {
        appState.addLog('info', 'Opening help documentation...');
        // In a real application, this would open a modal or redirect to documentation
        alert('Help documentation is under development. For now, refer to the API documentation at /api/docs');
      }
    };

    // Lifecycle
    onMounted(async () => {
      await businessLogic.initializeApp();
      appState.addLog('info', 'Application mounted and ready');

      // Auto-refresh status
      setInterval(() => {
        appState.lastUpdate.value = Utils.formatTime();
      }, 60000);
    });

    // Watch for theme changes to update charts
    watch(darkMode, () => {
      if (appState.result.value.rawData) {
        setTimeout(() => businessLogic.updateCharts(appState.result.value.rawData), 100);
      }
    });

    // Watch for mode changes
    watch(() => appState.mode.value, (newMode) => {
      appState.addLog('info', `Switched to ${newMode} mode`);
    });

    // Watch for active tab changes - render charts when charts tab becomes active
    watch(() => appState.activeTab.value, (newTab) => {
      if (newTab === 'charts' && appState.chartData.value.sofaTrend) {
        console.log('Charts tab activated, rendering stored chart data');
        // Small delay to ensure DOM is rendered
        setTimeout(() => businessLogic.renderCharts(), 100);
      }
    });

    // Return everything for template access
    return {
      // Theme
      darkMode,
      toggleTheme,

      // State (expose reactive refs directly)
      ...appState,

      // Computed
      ...computedProps,

      // Methods
      ...businessLogic,
      addLog: (type, message) => appState.addLog(type, message),
      clearLogs: () => appState.clearLogs(),
      logClass: (type) => appState.logClass(type)
    };
  }
});

// Mount the application
app.mount('#app');

// Export for debugging
if (typeof window !== 'undefined') {
  window.SepsisApp = {
    Utils,
    ChartManager,
    ApiClient,
    CONFIG
  };
}