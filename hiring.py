# Copy this into a new file called app.py and run with: streamlit run app.py
import streamlit as st
import pandas as pd
import xgboost as xgb

st.title("🚀 AI Recruitment Assistant")

st.sidebar.header("Candidate Input")
age = st.sidebar.slider("Age", 22, 45, 30)
exp = st.sidebar.slider("Experience (Years)", 0, 15, 5)
interview = st.sidebar.slider("Interview Score", 50, 95, 75)
skill = st.sidebar.slider("Skill Score", 40, 98, 80)
match_score = st.sidebar.slider("Match Score", 0.0, 1.0, 0.75)

input_data = pd.DataFrame([[age, exp, 1, interview, skill, 3, match_score]],
                          columns=['Age','ExperienceYears','EducationLevel','InterviewScore',
                                   'SkillScore','PreviousCompanies','Match_Score'])

if st.button("Predict"):
    # Load your trained models here in real deployment
    st.success("✅ Candidate is Likely to be Hired!")
    st.metric("Predicted Time to Hire", "45 days")
    st.metric("Match Score", f"{match_score:.2f}")
