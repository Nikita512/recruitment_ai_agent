#scoring.py
import re
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Read key from environment
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
client = None
if OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)

# --- Simple TF-IDF based score ---
def simple_tfidf_score(jd_text: str, resume_text: str) -> float:
    try:
        vec = TfidfVectorizer(stop_words='english').fit_transform([jd_text, resume_text])
        sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
        return float(sim) * 100
    except Exception as e:
        print("TF-IDF error:", e)
        return 0.0

# --- Embedding score using OpenAI embeddings ---
def embedding_score(jd_text: str, resume_text: str) -> float:
    if client is None:
        print("OpenAI client not configured; returning 0.0 embedding score.")
        return 0.0
    try:
        jd_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=jd_text
        ).data[0].embedding

        resume_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=resume_text
        ).data[0].embedding

        jd_vec = np.array(jd_embedding, dtype=float)
        resume_vec = np.array(resume_embedding, dtype=float)

        sim = float(np.dot(jd_vec, resume_vec) / (np.linalg.norm(jd_vec) * np.linalg.norm(resume_vec)))
        return sim * 100
    except Exception as e:
        print("Embedding error:", e)
        return 0.0

# --- Skill extraction helper ---
def extract_skills_from_text(text: str, skill_list: list) -> tuple[list, list]:
    """
    Returns (found_skills, missing_skills)
    """
    text_low = text.lower()
    found = [s for s in skill_list if re.search(r'\b' + re.escape(s.lower()) + r'\b', text_low)]
    missing = [s for s in skill_list if s not in found]
    return found, missing
