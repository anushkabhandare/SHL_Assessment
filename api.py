from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Dummy catalogue (same as used in your Streamlit app)
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

# Input model for API request
class RecommendationRequest(BaseModel):
    job_role: str
    skills: List[str]
    industry: str
    experience: str
    description: str

@app.post("/recommend")
def recommend_assessments(req: RecommendationRequest):
    input_text = f"{req.job_role} {' '.join(req.skills)} {req.industry} {req.experience} {req.description}"

    vectorizer = TfidfVectorizer()
    all_descriptions = catalogue_df['Description'].tolist() + [input_text]
    tfidf_matrix = vectorizer.fit_transform(all_descriptions)

    input_vector = tfidf_matrix[-1]
    assessment_vectors = tfidf_matrix[:-1]
    similarity_scores = cosine_similarity(input_vector, assessment_vectors).flatten()

    catalogue_df['Similarity Score'] = np.round(similarity_scores, 2)

    def generate_reason(description, input_skills):
        keywords = [skill for skill in input_skills if skill.lower() in description.lower()]
        if keywords:
            return f"Highly relevant due to focus on {', '.join(keywords)}."
        return "Matches job role and required competencies."

    top_matches = catalogue_df.sort_values(by='Similarity Score', ascending=False).head(5)
    results = [
        {
            "assessment_name": row['Assessment Name'],
            "description": row['Description'],
            "match_score": row['Similarity Score'],
            "why_recommended": generate_reason(row['Description'], req.skills)
        }
        for _, row in top_matches.iterrows()
    ]
    return {"recommendations": results}
