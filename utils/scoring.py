import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # small & fast

def simple_tfidf_score(jd_text, resume_text):
    vec = TfidfVectorizer(stop_words='english').fit_transform([jd_text, resume_text])
    sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    return float(sim) * 100

def embedding_score(jd_text, resume_text):
    emb = model.encode([jd_text, resume_text], convert_to_numpy=True, show_progress_bar=False)
    sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return float(sim) * 100

def extract_skills_from_text(text, skill_list):
    text_low = text.lower()
    found = []
    for s in skill_list:
        if re.search(r'\b' + re.escape(s.lower()) + r'\b', text_low):
            found.append(s)
    missing = [s for s in skill_list if s not in found]
    return found, missing