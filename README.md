# Healthcare-Diagnostic-AI
Clinical Diagnostic AI System 🩺
UHV Master's in Data Science NLP - Final Project

# Project Overview
This project is an advanced healthcare decision-support tool that uses Machine Learning to predict potential medical conditions based on natural language symptom descriptions. The system recognizes 140 different diseases and 175 unique symptoms, providing clinical descriptions and first-aid recommendations for each.

# Key Features
Natural Language Processing: Extracts clinical symptoms from messy user text.

Random Forest Classifier: A high-precision AI model trained on a custom healthcare dataset.

Severity Analysis: Automatically flags life-threatening symptom patterns (e.g., chest pain or difficulty breathing).

Clinical Knowledge Base: Integrated database for disease descriptions and precautions.

# Project Structure
/data: Contains the knowledge base (Training dataset, Severity weights, Disease Info, and Precautions).

/notebook2: Contains the application logic (final_project.py) and environment setup (requirements.txt).

# Technology Stack
Language: Python 3.x

Web Framework: Streamlit

Machine Learning: Scikit-Learn (Random Forest)

Data Handling: Pandas

NLP: SpaCy (en_core_web_sm)

# How to Run Locally
Clone the repository.

Install dependencies: pip install -r notebook2/requirements.txt

Run the app: streamlit run notebook2/final_project.py
