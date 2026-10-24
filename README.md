# Healthcare-Diagnostic-AI: Clinical Decision Support System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Model-Random%20Forest-orange.svg)](https://scikit-learn.org/)

An interactive clinical decision-support system (CDSS) designed to evaluate natural language patient symptom inputs, estimate condition probabilities, flag clinical severity tiers, and provide triage precautions.

---

## Executive Overview
Triage bottlenecks and delayed diagnostic escalation present persistent risks in acute care intake. This system processes free-text patient symptom reports, extracts clinical entities, maps them against symptom vectors, and deploys a multi-class Random Forest classifier trained on 140 diseases and 175 distinct symptoms.

### Core Capabilities
* **Clinical Symptom Ingestion:** Extracts and standardizes symptoms from unstructured descriptive phrases.
* **Predictive Diagnosis:** Ranks differential diagnostic probabilities using a multi-class Random Forest architecture.
* **Acuity & Severity Triage:** Incorporates structured symptom severity weights to alert for high-risk presentations (e.g., cardiopulmonary distress indicators).
* **Precautionary Decision Support:** Surfaces structured precautions and clinical background data aligned with differential predictions.

---

## System Architecture

```text
Healthcare-Diagnostic-AI/
│
├── data/
│   ├── disease_info.csv                       # Clinical disease descriptions
│   ├── disease_precaution.csv                 # Triage steps and care guidelines
│   ├── new_symptom_severity.csv               # Weighted acuity scoring
│   └── realistic_healthcare_symptom_dataset.csv # Primary training dataset
│
├── final_project.py                           # Core Streamlit app & ML pipeline
├── requirements.txt                           # Verified dependency specifications
└── README.md                                  # Technical documentation
