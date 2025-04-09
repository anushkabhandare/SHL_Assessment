# SHL Assessment Recommendation Engine (Enhanced Streamlit Version)

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Step 1: Load Product Catalogue
# --------------------------
data = {
    'Assessment Name': [
        'Cognitive Ability Test',
        'Situational Judgement Test - Sales',
        'Personality Profile - Leadership',
        'Technical Coding Assessment',
        'Customer Service Simulation',
        'Innovation & Creativity Assessment',
        'Data Interpretation & Reasoning',
        'Emotional Intelligence Assessment',
        'Analytical Thinking Test',
        'Project Management Simulation'
    ],
    'Description': [
        'Measures problem solving, numerical reasoning and logical thinking.',
        'Assesses decision-making and interpersonal skills in a sales context.',
        'Evaluates personality traits related to effective leadership.',
        'Tests programming skills in Python, Java and algorithms.',
        'Simulates customer service scenarios to assess communication and empathy.',
        'Identifies creative thinking and innovative problem-solving ability.',
        'Evaluates ability to interpret data, charts and make logical deductions.',
        'Assesses emotional regulation, self-awareness and empathy.',
        'Measures analytical thinking, logic-based decision making.',
        'Simulates real-world project management tasks under constraints.'
    ]
}

catalogue_df = pd.DataFrame(data)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="SHL Assessment Recommendation Engine", layout="centered")
st.title("🧠 SHL Assessment Recommendation Engine")
st.markdown("Use the dropdown menus below to get tailored SHL assessment suggestions based on your job role and profile.")

# Predefined options for interactivity
job_roles = ["Sales Manager", "Software Engineer", "Team Leader", "Customer Support", "Marketing Specialist", "Data Analyst", "HR Specialist", "Operations Manager"]
skill_options = ["communication", "negotiation", "leadership", "coding", "problem solving", "empathy", "creativity", "analytics", "project management", "data interpretation"]
industries = ["Retail", "Technology", "Finance", "Healthcare", "Education", "Telecom", "Manufacturing"]
experience_levels = ["Entry-Level", "Mid-Level", "Senior-Level", "Executive"]

# Input Fields with dropdowns
job_role = st.selectbox("Select Job Role", job_roles)
skills = st.multiselect("Select Key Skills", options=skill_options, default=["communication", "problem solving"])
industry = st.selectbox("Select Industry", industries)
experience_level = st.selectbox("Select Experience Level", experience_levels)

# Optional file uploader for custom job descriptions
st.markdown("### 📄 Upload a Custom Job Description (Optional)")
uploaded_file = st.file_uploader("Upload a .txt, .pdf, or .docx file with the job description", type=["txt", "pdf", "docx"], accept_multiple_files=False, help="Maximum file size: 5MB")

job_description_text = ""
if uploaded_file is not None:
    if uploaded_file.size <= 5 * 1024 * 1024:  # 5MB limit
        if uploaded_file.type == "text/plain":
            job_description_text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            job_description_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            import docx
            doc = docx.Document(uploaded_file)
            job_description_text = " ".join([para.text for para in doc.paragraphs])

        st.text_area("Preview Job Description", job_description_text, height=200)
    else:
        st.error("File size exceeds 5MB limit. Please upload a smaller file.")

import requests

if st.button("🎯 Get Assessment Recommendations"):
    st.info("Sending your data to the recommendation engine...")

    input_text = f"{job_role} {' '.join(skills)} {industry} {experience_level} {job_description_text}"
    payload = {"query": input_text}
    API_URL = "https://shl-fastapi-production.up.railway.app/recommend"

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            results = response.json()["recommended_assessments"]

            st.subheader("🔍 Top Recommended Assessments")
            for rec in results:
                st.markdown(f"### ✅ {rec['url'].split('/')[-2].replace('-', ' ').title()}")
                st.markdown(f"**Description:** {rec['description']}")
                st.markdown(f"**Duration:** {rec['duration']} mins")
                st.markdown(f"**Remote Support:** {rec['remote_support']}")
                st.markdown(f"**Adaptive Support:** {rec['adaptive_support']}")
                st.markdown(f"**Test Type:** {', '.join(rec['test_type'])}")
                st.markdown(f"[🔗 View Assessment]({rec['url']})")
                st.markdown("---")
        else:
            st.error(f"Failed to fetch recommendations. Status code: {response.status_code}")
    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")



    # Reason generation
    def generate_reason(description, input_skills):
        keywords = [skill for skill in input_skills if skill.lower() in description.lower()]
        if keywords:
            return f"Highly relevant due to focus on {', '.join(keywords)}."
        return "Matches job role and required competencies."

    recommendations['Why Recommended'] = recommendations.apply(
        lambda row: generate_reason(row['Description'], skill_list), axis=1
    )

    # Display Results
    st.subheader("🔍 Top Recommended Assessments")
    for _, row in recommendations.iterrows():
        st.markdown(f"### ✅ {row['Assessment Name']}")
        st.markdown(f"**Match Score:** {row['Similarity Score']:.2f}")
        st.markdown(f"**Description:** {row['Description']}")
        st.markdown(f"**Why Recommended:** {row['Why Recommended']}")
        st.markdown("---")

st.markdown("\n---\n")
st.caption("ℹ️ This tool is designed to assist recruiters and candidates in exploring SHL's assessment offerings tailored to specific job needs. Learn more at [shl.com](https://www.shl.com/en-in/resources/shl-labs/)")

