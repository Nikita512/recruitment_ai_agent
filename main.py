# main.py
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from utils.text_extract import extract_text_generic
from utils.scoring import embedding_score, extract_skills_from_text, simple_tfidf_score
from openai import OpenAI
import os, io, uuid, json
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure folders exist
os.makedirs("jds", exist_ok=True)
os.makedirs("resumes", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app = FastAPI(title="Recruitment AI Agent")
templates = Jinja2Templates(directory="templates")

# Initialize OpenAI client safely
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = None
if OPENAI_KEY:
    try:
        client = OpenAI(api_key=OPENAI_KEY)
    except Exception as e:
        print("⚠️ OpenAI init failed:", e)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Save JD text (from paste or upload)
@app.post("/upload_jd")
async def upload_jd(jd_text: str = Form(None), jd_file: UploadFile = File(None)):
    text = ""
    if jd_file:
        content = await jd_file.read()
        text = extract_text_generic(io.BytesIO(content), jd_file.filename)
    elif jd_text:
        text = jd_text

    if not text.strip():
        return {"error": "Empty JD provided."}

    jd_id = f"jd_{uuid.uuid4().hex[:8]}.txt"
    with open(os.path.join("jds", jd_id), "w", encoding="utf-8") as f:
        f.write(text)

    return {"status": "saved", "jd_file": jd_id}


# Upload resumes and generate scores
@app.post("/upload_resumes")
async def upload_resumes(resumes: List[UploadFile] = File(...)):
    results = []
    jd_files = sorted(os.listdir("jds"))
    if not jd_files:
        return {"error": "No JD found. Please upload or generate one first."}

    with open(os.path.join("jds", jd_files[-1]), "r", encoding="utf-8") as f:
        jd_text = f.read()

    common_skills = [
        "python", "fastapi", "sql", "aws", "docker", "kubernetes",
        "react", "java", "c++", "nlp", "machine learning", "flask"
    ]

    for resume in resumes[:10]:
        try:
            content = await resume.read()
            text = extract_text_generic(io.BytesIO(content), resume.filename)

            if not text.strip():
                raise ValueError("No readable text extracted from resume.")

            # TF-IDF score
            try:
                tf_score = simple_tfidf_score(jd_text, text)
            except Exception as e:
                print("TF-IDF scoring error:", e)
                tf_score = 0.0

            # Embedding score
            emb_score = 0.0
            if client:
                try:
                    emb_score = embedding_score(jd_text, text)
                except Exception as e:
                    print("Embedding scoring error:", e)

            final_score = (emb_score * 0.7) + (tf_score * 0.3)

            # Extract skills safely
            try:
                found, missing = extract_skills_from_text(text, common_skills)
            except Exception as e:
                print(f"Skill extraction failed for {resume.filename}: {e}")
                found, missing = [], []

            # Remarks
            if final_score < 40:
                remarks = "Not a strong match"
            elif final_score < 70:
                remarks = "Partial match"
            else:
                remarks = "Strong match"

            results.append({
                "filename": resume.filename,
                "score": round(float(final_score), 2),
                "found_skills": found,
                "missing_skills": missing,
                "remarks": remarks
            })

        except Exception as e:
            print(f"Error processing {resume.filename}: {e}")
            results.append({
                "filename": resume.filename,
                "score": 0,
                "found_skills": [],
                "missing_skills": [],
                "remarks": "Error processing file"
            })

    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    return {"results": results_sorted}


# Generate JD using LLM
@app.post("/generate_jd")
async def generate_jd(
    job_title: str = Form(...),
    years: str = Form(...),
    must_have: str = Form(""),
    company: str = Form(""),
    employment_type: str = Form("Full-time"),
    industry: str = Form(""),
    location: str = Form("")
):
    if not client:
        return {"error": "OpenAI client not configured. Set OPENAI_API_KEY."}

    prompt = f"""
    Generate a concise, professional job description for:
    Title: {job_title}
    Experience: {years} years
    Must-have skills: {must_have}
    Company: {company}
    Employment Type: {employment_type}
    Industry: {industry}
    Location: {location}
    Include key responsibilities and qualifications.
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=450,
        )
        jd_text = resp.choices[0].message.content
        jd_id = f"jd_{uuid.uuid4().hex[:8]}.txt"
        with open(os.path.join("jds", jd_id), "w", encoding="utf-8") as f:
            f.write(jd_text)
        return {"jd_file": jd_id, "jd_text": jd_text}

    except Exception as e:
        print("JD generation error:", e)
        return {"error": str(e)}


# Generate candidate email
@app.post("/generate_emails")
async def generate_emails(
    candidate_name: str = Form(...),
    candidate_email: str = Form(...),
    decision: str = Form("interview"),
    role: str = Form("")
):
    if not client:
        return {"error": "OpenAI client not configured. Set OPENAI_API_KEY."}

    if decision == "interview":
        prompt = f"Write a professional interview call email to {candidate_name} for the role {role}. Include subject and body."
    else:
        prompt = f"Write a polite rejection email to {candidate_name} for the role {role}. Include subject and body."

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        email_text = resp.choices[0].message.content
        return {"email": email_text}
    except Exception as e:
        print("Email generation error:", e)
        return {"error": str(e)}
