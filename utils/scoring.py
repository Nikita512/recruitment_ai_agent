import re
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# --- Initialize OpenAI client once ---
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

# --- Simple TF-IDF based score ---
def simple_tfidf_score(jd_text, resume_text):
    vec = TfidfVectorizer(stop_words='english').fit_transform([jd_text, resume_text])
    sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    return float(sim) * 100


# --- Embedding score using OpenAI (lightweight, no local model) ---
def embedding_score(jd_text, resume_text):
    try:
        jd_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=jd_text
        ).data[0].embedding

        resume_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=resume_text
        ).data[0].embedding

        jd_vec = np.array(jd_embedding)
        resume_vec = np.array(resume_embedding)

        sim = float(np.dot(jd_vec, resume_vec) / (np.linalg.norm(jd_vec) * np.linalg.norm(resume_vec)))
        return sim * 100
    except Exception as e:
        print("Embedding error:", e)
        return 0.0


# --- Skill extraction helper ---
def extract_skills_from_text(text, skill_list):
    text_low = text.lower()
    found = []
    for s in skill_list:
        if re.search(r'\b' + re.escape(s.lower()) + r'\b', text_low):
            found.append(s)
    missing = [s for s in skill_list if s not in found]
    return found, missing
