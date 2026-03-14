import streamlit as st
import pandas as pd
import os
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="Advanced Healthcare AI", page_icon="🩺", layout="wide")

# --- PATHS ---
base_path = r"C:\Users\farah\OneDrive\Desktop\Masters in data science\Natural Learning Processing\Labs\Final Project\data"

@st.cache_resource
def load_and_train():
    # 1. Load the 4 healthcare files using your EXACT uploaded filenames
    df = pd.read_csv(os.path.join(base_path, "realistic_healthcare_symptom_dataset.csv"))
    # Note: These files contain symptom data, but we use them as the primary lookup
    desc_df = pd.read_csv(os.path.join(base_path, "new_symptom_description.csv"))
    prec_df = pd.read_csv(os.path.join(base_path, "new_symptom_precaution.csv"))
    severity_df = pd.read_csv(os.path.join(base_path, "new_symptom_severity.csv"))
    
    # 2. Prepare Training Data
    X = df.drop('Disease', axis=1)
    y = df['Disease']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 3. Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    
    return model, le, X.columns.tolist(), desc_df, prec_df, severity_df

# --- INITIALIZE ---
try:
    model, le, symptom_cols, desc_df, prec_df, severity_df = load_and_train()
    st.sidebar.success("✅ Healthcare Engine Ready")
except Exception as e:
    st.error(f"Setup Error: {e}. Check if file names match exactly in the folder.")
    st.stop()

# --- SIDEBAR GLOSSARY ---
with st.sidebar:
    st.title("📚 Symptom Glossary")
    search = st.text_input("Search:", placeholder="e.g. fever")
    clean_display = sorted([s.replace('_', ' ') for s in symptom_cols])
    if search:
        st.write([s for s in clean_display if search.lower() in s])
    else:
        st.write(clean_display[:50])

# --- MAIN UI ---
st.title("🩺 Advanced Clinical Diagnostic AI")
user_input = st.text_area("How are you feeling?", height=150, placeholder="Example: I have a sore throat and fever.")

if st.button("Analyze Now"):
    if user_input:
        text = user_input.lower()
        
        # 1. FEATURE EXTRACTION
        input_vector = [0] * len(symptom_cols)
        found_symptoms = []
        total_severity = 0
        
        for i, col in enumerate(symptom_cols):
            clean_col = col.replace('_', ' ')
            if clean_col in text:
                input_vector[i] = 1
                found_symptoms.append(col)
                
                # Severity lookup
                sev_score = severity_df[severity_df['Symptom'].str.strip() == col.strip()]['Severity_Score'].values
                if len(sev_score) > 0:
                    total_severity += sev_score[0]

        if found_symptoms:
            # 2. PREDICTION
            probabilities = model.predict_proba([input_vector])[0]
            max_prob = max(probabilities)
            prediction = le.classes_[probabilities.argmax()]
            
            st.divider()
            
            # Severity Warning
            if total_severity > 7:
                st.error("🚨 **High Severity Warning:** Consult a doctor immediately.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.header(f"Likely Condition: {prediction}")
                st.write(f"**AI Confidence Score:** {max_prob*100:.1f}%")
                
                # FIX: Look up info based on the first detected symptom since your CSVs are symptom-based
                main_symptom = found_symptoms[0]
                desc_match = desc_df[desc_df['Symptom'].str.strip() == main_symptom.strip()]
                
                if not desc_match.empty:
                    st.info(f"**About your primary symptom ({main_symptom.replace('_',' ')}):**\n\n{desc_match['Description'].values[0]}")
                else:
                    st.info(f"Analysis complete for {prediction}. Please monitor your symptoms.")

            with col2:
                # FIX: Precaution Lookup based on detected symptoms
                prec_match = prec_df[prec_df['Symptom'].str.strip() == main_symptom.strip()]
                if not prec_match.empty:
                    st.subheader("🩹 Recommended Precautions")
                    for p_col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                        st.write(f"- {prec_match[p_col].values[0]}")
                else:
                    st.warning("General advice: Rest, hydrate, and monitor your health.")
            
            st.write(f"**Detected Symptoms:** {', '.join([s.replace('_', ' ') for s in found_symptoms])}")
        else:
            st.error("No symptoms recognized. Please refer to the glossary in the sidebar.")

st.divider()
st.caption("UHV NLP Final Project - Clinical Decision Support System")