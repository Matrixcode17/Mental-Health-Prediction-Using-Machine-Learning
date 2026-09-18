import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Page Configuration & Styling
# --------------------------------------------------
st.set_page_config(
    page_title="Workplace Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .result-box-positive {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1.5rem;
    }
    .result-box-negative {
        background-color: #F0FDF4;
        border-left: 5px solid #22C55E;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model & Mappings
# --------------------------------------------------
@st.cache_resource
def load_pipeline():
    try:
        data = joblib.load("mental_health_model.pkl")
        return data['model'], data['label_dict'], data['features']
    except Exception as e:
        st.error(f"Detailed Error loading model: {e}")
        raise e  # This will display the full traceback in your terminal and browser

model, label_dict, feature_columns = load_pipeline()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown('<div class="main-title">Workplace Mental Health Predictor 🧠</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Assess workplace risk factors and treatment propensity using machine learning (Optimized via AdaBoost for high recall).</div>', 
    unsafe_allow_html=True
)

if model is None:
    st.warning("⚠️ `mental_health_model.pkl` not found. Please train the model and save the artifact file in the root directory.")
    st.stop()

# --------------------------------------------------
# Input Form UI
# --------------------------------------------------
with st.form(key="prediction_form"):
    st.markdown("### 1. Demographic & Work Environment")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gender_raw = st.selectbox("Gender", options=["Male", "Female", "Other"])
    with col2:
        self_employed = st.selectbox("Self Employed?", options=label_dict['label_self_employed'])
    with col3:
        remote_work = st.selectbox("Remote Work?", options=label_dict['label_remote_work'])
    with col4:
        tech_company = st.selectbox("Tech Company?", options=label_dict['label_tech_company'])

    st.markdown("---")
    st.markdown("### 2. Mental Health & Well-being Indicators")
    col5, col6, col7 = st.columns(3)

    with col5:
        family_history = st.selectbox("Family History of Mental Illness?", options=label_dict['label_family_history'])
    with col6:
        work_interfere = st.selectbox("Mental Health Interferes with Work?", options=label_dict['label_work_interfere'])
    with col7:
        obs_consequence = st.selectbox("Observed Negative Consequences for Coworkers?", options=label_dict['label_obs_consequence'])

    st.markdown("---")
    st.markdown("### 3. Company Policies, Benefits & Openness")
    col8, col9, col10 = st.columns(3)

    with col8:
        benefits = st.selectbox("Employer Mental Health Benefits", options=label_dict['label_benefits'])
        care_options = st.selectbox("Aware of Available Care Options?", options=label_dict['label_care_options'])
        wellness_program = st.selectbox("Wellness Program Offered?", options=label_dict['label_wellness_program'])
        seek_help = st.selectbox("Employer Resources to Seek Help?", options=label_dict['label_seek_help'])

    with col9:
        anonymity = st.selectbox("Anonymity Protected if Using Services?", options=label_dict['label_anonymity'])
        leave = st.selectbox("Ease of Taking Medical Leave for Mental Health", options=label_dict['label_leave'])
        mental_health_consequence = st.selectbox("Negative Consequence Discussing Mental Health?", options=label_dict['label_mental_health_consequence'])
        phys_health_consequence = st.selectbox("Negative Consequence Discussing Physical Health?", options=label_dict['label_phys_health_consequence'])

    with col10:
        coworkers = st.selectbox("Willing to Discuss with Coworkers?", options=label_dict['label_coworkers'])
        supervisor = st.selectbox("Willing to Discuss with Direct Supervisor?", options=label_dict['label_supervisor'])
        mental_health_interview = st.selectbox("Bring Up Mental Health in an Interview?", options=label_dict['label_mental_health_interview'])
        phys_health_interview = st.selectbox("Bring Up Physical Health in an Interview?", options=label_dict['label_phys_health_interview'])
        mental_vs_physical = st.selectbox("Does Employer Take Mental Health as Seriously as Physical?", options=label_dict['label_mental_vs_physical'])

    submitted = st.form_submit_button("Predict Treatment Propensity", use_container_width=True)

# --------------------------------------------------
# Preprocessing & Inference
# --------------------------------------------------
if submitted:
    # 1. Standardize Gender text (matching notebook logic)
    g_lower = gender_raw.lower()
    if ('f' in g_lower) or ('woman' in g_lower) and ('tr' not in g_lower):
        clean_gender = 'female'
    elif ('m' in g_lower) or ('guy' in g_lower) and ('tr' not in g_lower):
        clean_gender = 'male'
    else:
        clean_gender = 'other'

    # 2. Construct raw dictionary
    raw_inputs = {
        'Gender': clean_gender,
        'self_employed': self_employed,
        'family_history': family_history,
        'work_interfere': work_interfere,
        'remote_work': remote_work,
        'tech_company': tech_company,
        'benefits': benefits,
        'care_options': care_options,
        'wellness_program': wellness_program,
        'seek_help': seek_help,
        'anonymity': anonymity,
        'leave': leave,
        'mental_health_consequence': mental_health_consequence,
        'phys_health_consequence': phys_health_consequence,
        'coworkers': coworkers,
        'supervisor': supervisor,
        'mental_health_interview': mental_health_interview,
        'phys_health_interview': phys_health_interview,
        'mental_vs_physical': mental_vs_physical,
        'obs_consequence': obs_consequence,
    }

    # 3. Encode inputs based on labelDict classes_ from training
    encoded_features = {}
    for col, val in raw_inputs.items():
        classes = list(label_dict['label_' + col])
        encoded_features[col] = classes.index(val) if val in classes else 0

    input_df = pd.DataFrame([encoded_features])
    input_df = input_df[feature_columns]  # Enforce correct training feature order

    # 4. Predict
    pred_code = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]
    
    treatment_labels = list(label_dict['label_treatment'])
    pred_label = treatment_labels[pred_code]
    treatment_prob = prob[treatment_labels.index("Yes")] if "Yes" in treatment_labels else prob[1]

    # 5. Display Result
    if pred_label == "Yes":
        st.markdown(
            f"""
            <div class="result-box-positive">
                <h3 style="color: #991B1B; margin: 0;">Predicted Outcome: Treatment Recommended (Yes)</h3>
                <p style="color: #475569; margin-top: 0.5rem;">
                    The model identifies significant risk indicators (Work Interference, Benefits gap, or Family History) that suggest professional support or clinical consultation is warranted.
                </p>
                <b>Probability Score:</b> {treatment_prob * 100:.1f}%
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="result-box-negative">
                <h3 style="color: #166534; margin: 0;">Predicted Outcome: Treatment Not Currently Indicated (No)</h3>
                <p style="color: #475569; margin-top: 0.5rem;">
                    Current indicators do not show elevated propensity for immediate mental health treatment. Continuing access to open workplace resources and supportive environments is recommended.
                </p>
                <b>Confidence Score:</b> {(1 - treatment_prob) * 100:.1f}%
            </div>
            """, 
            unsafe_allow_html=True
        )