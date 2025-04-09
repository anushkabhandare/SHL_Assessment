# SHL API (Spec Compliant)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

@app.get("/")
def root():
    return {"message": "SHL Recommendation API is working!"}

# Sample SHL assessment catalogue with required metadata
data = {
    'Assessment Name': [
        'Cognitive Ability Test',
        'Situational Judgement Test - Sales',
        'Personality Profile - Leadership',
        'Technical Coding Assessment',
        'Customer Service Simulation'
    ],
    'Description': [
        'Measures problem solving, numerical reasoning and logical thinking.',
        'Assesses decision-making and interpersonal skills in a sales context.',
        'Evaluates personality traits related to effective leadership.',
        'Tests programming skills in Python, Java and algorithms.',
        'Simulates customer service scenarios to assess communication and empathy.'
    ],
    'URL': [
        'https://www.shl.com/cognitive',
        'https://www.shl.com/sjt-sales',
        'https://www.shl.com/personality-leadership',
        'https://www.shl.com/tech-coding',
        'https://www.shl.com/customer-service'
    ],
    'Adaptive Support': ['Yes', 'No', 'Yes', 'No', 'No'],
    'Duration': [40, 25, 30, 60, 35],
    'Remote Support': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
    'Test Type': [
        ['Cognitive Ability'],
        ['Situational Judgment'],
        ['Personality'],
        ['Knowledge & Skills'],
        ['Simulation']
    ]
}

catalogue_df = pd.DataFrame(data)

class RecommendInput(BaseModel):
    query: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}



@app.post("/recommend")
def recommend(input: RecommendInput):
    input_text = input.query
    vectorizer = TfidfVectorizer()
    all_descriptions = catalogue_df['Description'].tolist() + [input_text]
    tfidf_matrix = vectorizer.fit_transform(all_descriptions)

    input_vector = tfidf_matrix[-1]
    assessment_vectors = tfidf_matrix[:-1]
    similarity_scores = cosine_similarity(input_vector, assessment_vectors).flatten()

    catalogue_df['Similarity Score'] = np.round(similarity_scores, 2)
    top_matches = catalogue_df.sort_values(by='Similarity Score', ascending=False).head(10)

    results = []
    for _, row in top_matches.iterrows():
        results.append({
            "url": row['URL'],
            "adaptive_support": row['Adaptive Support'],
            "description": row['Description'],
            "duration": row['Duration'],
            "remote_support": row['Remote Support'],
            "test_type": row['Test Type']
        })

    return {"recommended_assessments": results}
