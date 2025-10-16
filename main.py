# main.py
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from utils.text_extract import extract_text_generic
from utils.scoring import embedding_score, extract_skills_from_text, simple_tfidf_score
import os, textwrap, json, io
import openai
from typing import List
import uuid
from dotenv import load_dotenv

# Load env vars from .env if present (useful locally)
load_dotenv()

# Ensure folders exist (Render may not create them)
os.makedirs("jds", exist_ok=True)
os.makedirs("resumes", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app = FastAPI(title='Recruitment AI Agent')
templates = Jinja2Templates(directory='templates')

# Ensure OpenAI key is available for OpenAI calls
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

# save JD text (from paste or upload)
@app.post('/upload_jd')
async def upload_jd(jd_text: str = Form(None), jd_file: UploadFile = File(None)):
    text = ''
    if jd_file:
        content = await jd_file.read()
        text = extract_text_generic(io.BytesIO(content), jd_file.filename)
    else:
        text = jd_text or ''
    # store JD
    jid = f"jd_{uuid.uuid4().hex[:8]}.txt"
    with open(os.path.join('jds', jid), 'w', encoding='utf-8') as f:
        f.write(text)
    return {'status':'saved', 'jd_file': jid}

# upload resumes and score
@app.post('/upload_resumes')
async def upload_resumes(resumes: List[UploadFile] = File(...)):
    results = []
    # load latest JD: simple approach -- pick the last file in jds folder
    jd_files = sorted(os.listdir('jds'))
    if not jd_files:
        return {'error':'No JD found; please add a JD first.'}
    with open(os.path.join('jds', jd_files[-1]), 'r', encoding='utf-8') as f:
        jd_text = f.read()

    # simple skill list heuristic
    common_skills = ['python','fastapi','sql','aws','docker','kubernetes','react','java','c++','nlp','machine learning']

    for r in resumes[:10]:
        content = await r.read()
        text = extract_text_generic(io.BytesIO(content), r.filename)
        # score with embedding & TF-IDF averaged
        try:
            emb_score = embedding_score(jd_text, text)
        except Exception as e:
            print("Embedding scoring error:", e)
            emb_score = 0.0
        try:
            tf_score = simple_tfidf_score(jd_text, text)
        except Exception as e:
            print("TF-IDF scoring error:", e)
            tf_score = 0.0
        final_score = (emb_score * 0.7) + (tf_score * 0.3)
        found, missing = extract_skills_from_text(text, common_skills)
        remarks = []
        if final_score < 40:
            remarks.append('Not a strong match')
        elif final_score < 70:
            remarks.append('Partial match')
        else:
            remarks.append('Strong match')
        results.append({
            'filename': r.filename,
            'score': round(float(final_score),2),
            'found_skills': found,
            'missing_skills': missing,
            'remarks': remarks
        })
    # sort by score desc
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    return {'results': results_sorted}

# Generate JD via LLM (simple endpoint)
@app.post('/generate_jd')
async def generate_jd(job_title: str = Form(...), years: str = Form(...), must_have: str = Form(''), company: str = Form(''), employment_type: str = Form('Full-time'), industry: str = Form(''), location: str = Form('')):
    prompt = f"""Generate a concise, professional job description for:
Title: {job_title}
Years: {years}
Must-have skills (comma-separated): {must_have}
Company: {company}
Employment Type: {employment_type}
Industry: {industry}
Location: {location}
Provide a JD including responsibilities and required qualifications."""

    # Use OpenAI completion (user must set OPENAI_API_KEY env var)
    if not openai.api_key:
        return {'error':'OPENAI_API_KEY not set in environment. Set it before calling this endpoint.'}
    # Choose a stable model name; adjust if you have access to other models
    resp = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=[{'role':'user','content':prompt}],
        max_tokens=450
    )
    jd_text = resp['choices'][0]['message']['content']
    jid = f"jd_{uuid.uuid4().hex[:8]}.txt"
    with open(os.path.join('jds', jid), 'w', encoding='utf-8') as f:
        f.write(jd_text)
    return {'jd_file': jid, 'jd_text': jd_text}

# Generate email templates using LLM
@app.post('/generate_emails')
async def generate_emails(candidate_name: str = Form(...), candidate_email: str = Form(...), decision: str = Form('interview'), role: str = Form('')):
    if not openai.api_key:
        return {'error':'OPENAI_API_KEY not set.'}
    if decision == 'interview':
        prompt = f"Write a professional interview call email to {candidate_name} for the role {role}. Provide subject and body."
    else:
        prompt = f"Write a polite rejection email to {candidate_name} for the role {role}. Provide subject and body."
    resp = openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}], max_tokens=300)
    return {'email': resp['choices'][0]['message']['content']}
