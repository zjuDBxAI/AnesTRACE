// Published aggregate results from arXiv:2609.32740v1, Tables 3 and 4.
// Scores are percentages; latency is the reported p95 in seconds.
window.ANESTRACE_DATA = {
  version: "arXiv:2609.32740v1",
  published: "2026-09-26",
  source: "https://arxiv.org/pdf/2609.32740v1",
  categories: {
    proprietary: "Proprietary",
    general: "General-purpose",
    medical: "Medical-domain"
  },
  metrics: {
    perception: [
      { key: "teeFoundational", label: "Foundational perception", unit: "ACC", group: "TEE", better: "high" },
      { key: "teeGroundAcc", label: "Visual grounding", unit: "ACC", group: "TEE", better: "high" },
      { key: "teeGroundMiou", label: "Visual grounding", unit: "mIoU", group: "TEE", better: "high" },
      { key: "teeFunctionalAcc", label: "Functional assessment", unit: "ACC", group: "TEE", better: "high" },
      { key: "teeFactualF1", label: "Functional assessment", unit: "Factual-F1", group: "TEE", better: "high" },
      { key: "waveValueAcc", label: "Value extraction", unit: "ACC", group: "Waveform", better: "high" },
      { key: "waveTrendF1", label: "Trend reasoning", unit: "Set-F1", group: "Waveform", better: "high" },
      { key: "waveAnomalyF1", label: "Anomaly perception", unit: "Metric-F1", group: "Waveform", better: "high" }
    ],
    single: [
      { key: "risk", label: "Risk", unit: "score", group: "Task scores", better: "high" },
      { key: "diagnosis", label: "Diagnosis", unit: "score", group: "Task scores", better: "high" },
      { key: "intervention", label: "Intervention", unit: "score", group: "Task scores", better: "high" },
      { key: "reassessment", label: "Reassessment", unit: "score", group: "Task scores", better: "high" },
      { key: "avg", label: "Average", unit: "score", group: "Aggregate & efficiency", better: "high" },
      { key: "l2Safety", label: "Safety error", unit: "%", group: "Aggregate & efficiency", better: "low" },
      { key: "l2Latency", label: "P95 latency", unit: "s", group: "Aggregate & efficiency", better: "low", digits: 2 }
    ],
    multi: [
      { key: "turnAvg", label: "Turn-level average", unit: "score", group: "Clinical quality", better: "high" },
      { key: "temporal", label: "Temporal consistency", unit: "score", group: "Clinical quality", better: "high" },
      { key: "l3Safety", label: "Safety error", unit: "%", group: "Safety & agent behavior", better: "low" },
      { key: "evidence", label: "Evidence acquisition", unit: "%", group: "Safety & agent behavior", better: "high" },
      { key: "l3Latency", label: "P95 latency", unit: "s", group: "Safety & agent behavior", better: "low" }
    ]
  },
  perception: [
    { model: "GPT-6-Astra", category: "proprietary", teeFoundational: 73.9, teeGroundAcc: 62.9, teeGroundMiou: 32.2, teeFunctionalAcc: 40.6, teeFactualF1: 27.5, waveValueAcc: 91.5, waveTrendF1: 61.1, waveAnomalyF1: 64.8 },
    { model: "Gemini 3 Pro Preview", category: "proprietary", teeFoundational: 71.6, teeGroundAcc: 46.1, teeGroundMiou: 24.6, teeFunctionalAcc: 34.4, teeFactualF1: 23.7, waveValueAcc: 77.0, waveTrendF1: 62.6, waveAnomalyF1: 54.8 },
    { model: "Claude-Fable-5", category: "proprietary", teeFoundational: 74.5, teeGroundAcc: 42.7, teeGroundMiou: 23.4, teeFunctionalAcc: 37.8, teeFactualF1: 26.9, waveValueAcc: 80.5, waveTrendF1: 68.7, waveAnomalyF1: 59.6 },
    { model: "Qwen3.8-MAX", category: "general", teeFoundational: 73.2, teeGroundAcc: 51.7, teeGroundMiou: 22.8, teeFunctionalAcc: 48.2, teeFactualF1: 29.9, waveValueAcc: 72.0, waveTrendF1: 69.6, waveAnomalyF1: 42.8 },
    { model: "Qwen3.8-27B", category: "general", teeFoundational: 58.5, teeGroundAcc: 33.7, teeGroundMiou: 14.8, teeFunctionalAcc: 44.5, teeFactualF1: 14.0, waveValueAcc: 57.5, waveTrendF1: 72.5, waveAnomalyF1: 31.3 },
    { model: "Qwen3.5-27B", category: "general", teeFoundational: 58.5, teeGroundAcc: 43.8, teeGroundMiou: 18.9, teeFunctionalAcc: 43.2, teeFactualF1: 16.5, waveValueAcc: 56.0, waveTrendF1: 65.3, waveAnomalyF1: 36.5 },
    { model: "Qwen3.5-9B", category: "general", teeFoundational: 54.2, teeGroundAcc: 52.8, teeGroundMiou: 15.5, teeFunctionalAcc: 42.0, teeFactualF1: 8.7, waveValueAcc: 54.5, waveTrendF1: 60.7, waveAnomalyF1: 33.1 },
    { model: "Qwen3-VL-8B", category: "general", teeFoundational: 51.5, teeGroundAcc: 23.6, teeGroundMiou: 14.4, teeFunctionalAcc: 35.6, teeFactualF1: 5.7, waveValueAcc: 45.0, waveTrendF1: 58.2, waveAnomalyF1: 23.5 },
    { model: "LLaVA-OneVision-2-8B", category: "general", teeFoundational: 49.9, teeGroundAcc: 50.6, teeGroundMiou: 14.3, teeFunctionalAcc: 37.1, teeFactualF1: 9.7, waveValueAcc: 49.5, waveTrendF1: 60.6, waveAnomalyF1: 23.6 },
    { model: "InternVL3.5-8B", category: "general", teeFoundational: 54.0, teeGroundAcc: 48.3, teeGroundMiou: 13.9, teeFunctionalAcc: 34.8, teeFactualF1: 6.3, waveValueAcc: 41.0, waveTrendF1: 70.1, waveAnomalyF1: 17.3 },
    { model: "Fleming-VL-38B", category: "medical", teeFoundational: 50.3, teeGroundAcc: 37.1, teeGroundMiou: 14.8, teeFunctionalAcc: 28.2, teeFactualF1: 11.2, waveValueAcc: 41.5, waveTrendF1: 59.4, waveAnomalyF1: 24.0 },
    { model: "Lingshu-32B", category: "medical", teeFoundational: 55.0, teeGroundAcc: 46.1, teeGroundMiou: 12.1, teeFunctionalAcc: 35.2, teeFactualF1: 17.6, waveValueAcc: 49.5, waveTrendF1: 64.6, waveAnomalyF1: 20.0 },
    { model: "Lingshu-I-8B", category: "medical", teeFoundational: 53.5, teeGroundAcc: 56.2, teeGroundMiou: 13.5, teeFunctionalAcc: 33.4, teeFactualF1: 9.2, waveValueAcc: 39.0, waveTrendF1: 59.5, waveAnomalyF1: 23.3 },
    { model: "Fleming-VL-8B", category: "medical", teeFoundational: 47.2, teeGroundAcc: 46.1, teeGroundMiou: 8.7, teeFunctionalAcc: 12.9, teeFactualF1: 7.3, waveValueAcc: 38.5, waveTrendF1: 41.1, waveAnomalyF1: 17.6 }
  ],
  decision: [
    { model: "GPT-6-Astra", category: "proprietary", risk: 80.8, diagnosis: 93.7, intervention: 96.0, reassessment: 98.4, avg: 92.2, l2Safety: 3.4, l2Latency: 33.41, turnAvg: 82.7, temporal: 87.6, l3Safety: 17.5, evidence: 61.8, l3Latency: 74.3 },
    { model: "Gemini 3 Pro Preview", category: "proprietary", risk: 71.1, diagnosis: 74.7, intervention: 68.9, reassessment: 86.1, avg: 75.2, l2Safety: 13.5, l2Latency: 26.95, turnAvg: 66.1, temporal: 78.0, l3Safety: 38.5, evidence: 74.4, l3Latency: 101.4 },
    { model: "Claude-Fable-5", category: "proprietary", risk: 87.1, diagnosis: 91.9, intervention: 87.8, reassessment: 96.3, avg: 90.8, l2Safety: 9.4, l2Latency: 39.96, turnAvg: 79.4, temporal: 82.5, l3Safety: 24.3, evidence: 84.4, l3Latency: 69.2 },
    { model: "DeepSeek-V4-Pro", category: "general", risk: 78.2, diagnosis: 83.5, intervention: 79.0, reassessment: 93.8, avg: 83.6, l2Safety: 14.8, l2Latency: 14.04, turnAvg: 80.0, temporal: 81.9, l3Safety: 19.2, evidence: 89.7, l3Latency: 14.4 },
    { model: "Qwen3.8-Max", category: "general", risk: 70.7, diagnosis: 79.6, intervention: 71.6, reassessment: 87.5, avg: 77.4, l2Safety: 15.1, l2Latency: 23.51, turnAvg: 67.1, temporal: 85.3, l3Safety: 40.1, evidence: 62.8, l3Latency: 22.9 },
    { model: "Qwen3.5-27B", category: "general", risk: 69.4, diagnosis: 74.6, intervention: 70.2, reassessment: 77.9, avg: 73.0, l2Safety: 28.6, l2Latency: 22.14, turnAvg: 69.5, temporal: 81.6, l3Safety: 32.5, evidence: 83.3, l3Latency: 34.4 },
    { model: "Qwen3.8-27B", category: "general", risk: 71.4, diagnosis: 75.5, intervention: 66.1, reassessment: 75.0, avg: 72.0, l2Safety: 21.5, l2Latency: 22.94, turnAvg: 66.1, temporal: 69.2, l3Safety: 34.1, evidence: 86.9, l3Latency: 39.9 },
    { model: "GPT-OSS-20B", category: "general", risk: 69.1, diagnosis: 65.8, intervention: 50.7, reassessment: 66.1, avg: 62.9, l2Safety: 37.5, l2Latency: 17.90, turnAvg: 57.1, temporal: 67.2, l3Safety: 53.4, evidence: 59.5, l3Latency: 106.6 },
    { model: "GLM-4.7-Flash", category: "general", risk: 58.5, diagnosis: 58.6, intervention: 46.1, reassessment: 59.7, avg: 55.7, l2Safety: 48.2, l2Latency: 36.62, turnAvg: 44.1, temporal: 49.9, l3Safety: 70.8, evidence: 95.4, l3Latency: 63.6 },
    { model: "Qwen3.5-9B", category: "general", risk: 65.3, diagnosis: 70.0, intervention: 57.0, reassessment: 71.9, avg: 66.1, l2Safety: 27.7, l2Latency: 15.18, turnAvg: 55.1, temporal: 64.2, l3Safety: 61.9, evidence: 89.5, l3Latency: 27.2 },
    { model: "Qwen3-8B", category: "general", risk: 58.2, diagnosis: 52.2, intervention: 37.6, reassessment: 50.8, avg: 49.7, l2Safety: 57.6, l2Latency: 24.50, turnAvg: 30.2, temporal: 45.2, l3Safety: 64.1, evidence: 32.7, l3Latency: 10.3 },
    { model: "Morpheus-32B", category: "medical", risk: 55.0, diagnosis: 50.4, intervention: 41.7, reassessment: 58.0, avg: 51.3, l2Safety: 56.9, l2Latency: 20.31, turnAvg: 46.2, temporal: 56.0, l3Safety: 66.0, evidence: 67.5, l3Latency: 86.9 },
    { model: "Fleming-R1-32B", category: "medical", risk: 63.9, diagnosis: 63.9, intervention: 46.4, reassessment: 67.0, avg: 60.3, l2Safety: 37.6, l2Latency: 14.55, turnAvg: 52.8, temporal: 66.9, l3Safety: 56.5, evidence: 11.3, l3Latency: 22.8 },
    { model: "HuatuoGPT-3-32B", category: "medical", risk: 62.7, diagnosis: 60.7, intervention: 47.6, reassessment: 59.7, avg: 57.7, l2Safety: 50.9, l2Latency: 25.05, turnAvg: 49.9, temporal: 61.7, l3Safety: 64.7, evidence: 73.4, l3Latency: 31.3 },
    { model: "Baichuan-M2-32B", category: "medical", risk: 57.9, diagnosis: 62.0, intervention: 48.2, reassessment: 51.5, avg: 54.9, l2Safety: 42.1, l2Latency: 75.10, turnAvg: 49.5, temporal: 56.1, l3Safety: 66.7, evidence: 96.0, l3Latency: 33.0 },
    { model: "MedGemma-27B-text-it", category: "medical", risk: 64.7, diagnosis: 61.2, intervention: 41.5, reassessment: 63.3, avg: 57.7, l2Safety: 43.7, l2Latency: 12.44, turnAvg: 48.7, temporal: 54.3, l3Safety: 59.4, evidence: 40.5, l3Latency: 34.4 },
    { model: "HuatuoGPT-3-8B", category: "medical", risk: 44.3, diagnosis: 38.8, intervention: 35.6, reassessment: 45.5, avg: 41.1, l2Safety: 85.9, l2Latency: 9.87, turnAvg: 33.8, temporal: 48.2, l3Safety: 86.6, evidence: 29.1, l3Latency: 99.9 },
    { model: "Morpheus-7B", category: "medical", risk: 47.0, diagnosis: 42.3, intervention: 39.0, reassessment: 50.4, avg: 44.7, l2Safety: 67.4, l2Latency: 26.80, turnAvg: 35.1, temporal: 46.7, l3Safety: 78.2, evidence: 8.3, l3Latency: 50.4 },
    { model: "Fleming-R1-7B", category: "medical", risk: 55.0, diagnosis: 52.5, intervention: 40.5, reassessment: 55.2, avg: 50.8, l2Safety: 59.0, l2Latency: 9.19, turnAvg: 44.5, temporal: 56.6, l3Safety: 69.4, evidence: 4.6, l3Latency: 33.1 }
  ]
};
