import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="Advanced Healthcare AI", page_icon="🩺", layout="wide")

# --- RELATIVE PATH LOGIC ---
# This looks at where this file is (notebook2), goes up one level, and finds the 'data' folder.
# This is the "secret ingredient" for making it work on the web.
base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

@st.cache_resource
def load_and_train():
    # 1. Load the 4 healthcare files from the data folder
    df = pd.read_csv(os.path.join(base_path, "realistic_healthcare_symptom_dataset.csv"))
    desc_df = pd.read_csv(os.path.join(base_path, "disease_info.csv"))
    prec_df = pd.read_csv(os.path.join(base_path, "disease_precaution.csv"))
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

# --- INITIALIZE ENGINE ---
try:
    model, le, symptom_cols, desc_df, prec_df, severity_df = load_and_train()
    st.sidebar.success("✅ Healthcare Engine Ready")
except Exception as e:
    st.error(f"Setup Error: {e}. Please check your 'data' folder path.")
    st.stop()

# --- SIDEBAR GLOSSARY ---
with st.sidebar:
    st.title("📚 Symptom Glossary")
    st.markdown("Words the AI recognizes:")
    search = st.text_input("Search:", placeholder="e.g. fever")
    clean_display = sorted([s.replace('_', ' ') for s in symptom_cols])
    if search:
        st.write([s for s in clean_display if search.lower() in s])
    else:
        st.write(clean_display[:30] + ["...and more"])

# --- MAIN UI ---
st.title("🩺 Advanced Clinical Diagnostic AI")
st.markdown("### Describe your symptoms in natural language")

user_input = st.text_area("How are you feeling today?", height=150, placeholder="Example: I have a persistent cough, sore throat, and a high fever.")

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
                
                # Severity Calculation
                sev_match = severity_df[severity_df['Symptom'].str.strip() == col.strip()]['Severity_Score'].values
                if len(sev_match) > 0:
                    total_severity += sev_match[0]

        if found_symptoms:
            # 2. PREDICTION
            probabilities = model.predict_proba([input_vector])[0]
            max_prob = max(probabilities)
            prediction = le.classes_[probabilities.argmax()]
            
            st.divider()
            
            # Severity Flagging
            if total_severity > 10:
                st.error("🚨 **High Severity Warning:** These symptoms may require urgent medical attention. Please consult a professional immediately.")
            elif total_severity > 5:
                st.warning("⚠️ **Moderate Severity:** Consider scheduling a doctor's visit soon.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.header(f"Likely Condition: {prediction}")
                st.write(f"**AI Confidence Score:** {max_prob*100:.1f}%")
                
                # DISEASE DESCRIPTION LOOKUP
                # Matches predicted disease to the disease_info.csv
                desc_match = desc_df[desc_df['Disease'].str.strip().str.lower() == prediction.lower().strip()]
                if not desc_match.empty:
                    st.info(f"**About this condition:**\n\n{desc_match['Description'].values[0]}")
                else:
                    st.info("Additional clinical data for this condition is currently being integrated.")

            with col2:
                # PRECAUTION LOOKUP
                # Matches predicted disease to the disease_precaution.csv
                prec_match = prec_df[prec_df['Disease'].str.strip().str.lower() == prediction.lower().strip()]
                if not prec_match.empty:
                    st.subheader("🩹 Recommended First Aid")
                    for p_col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                        val = prec_match[p_col].values[0]
                        if pd.notna(val):
                            st.write(f"- {val}")
                else:
                    st.warning("General advice: Rest, maintain hydration, and monitor your vitals.")
            
            st.write(f"**Symptoms Detected:** {', '.join([s.replace('_', ' ') for s in found_symptoms])}")
        else:
            st.error("I couldn't identify specific symptoms. Please try rephrasing or use terms from the glossary.")

st.divider()
st.caption("UHV Masters Data Science NLP Final Project - Clinical Decision Support System")