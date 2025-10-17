from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from utils.text_extract import extract_text_generic
from utils.scoring import embedding_score, extract_skills_from_text, simple_tfidf_score
import os, io, uuid
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Ensure folders exist
os.makedirs("jds", exist_ok=True)
os.makedirs("resumes", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app = FastAPI(title="Recruitment AI Agent")
templates = Jinja2Templates(directory="templates")

# OpenAI client
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# ----------------------
# Home page
# ----------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ----------------------
# JD Upload
# ----------------------
@app.post("/upload_jd")
async def upload_jd(jd_text: str = Form(None), jd_file: UploadFile = File(None)):
    try:
        text = ""
        if jd_file:
            content = await jd_file.read()
            text = extract_text_generic(io.BytesIO(content), jd_file.filename)
        else:
            text = jd_text or ""

        jid = f"jd_{uuid.uuid4().hex[:8]}.txt"
        with open(os.path.join("jds", jid), "w", encoding="utf-8") as f:
            f.write(text)

        return JSONResponse({"status": "saved", "jd_file": jid, "message": "JD saved successfully!"})
    except Exception as e:
        return JSONResponse({"error": f"Failed to upload JD: {str(e)}"})

# ----------------------
# Resume Upload & Scoring
# ----------------------
@app.post("/upload_resumes")
async def upload_resumes(resumes: List[UploadFile] = File(...)):
    try:
        results = []

        # Load latest JD
        jd_files = sorted(os.listdir("jds"))
        if not jd_files:
            return JSONResponse({"error": "No JD found; please add a JD first."})
        with open(os.path.join("jds", jd_files[-1]), "r", encoding="utf-8") as f:
            jd_text = f.read()

        # Common skill list
        common_skills = ["python","fastapi","sql","aws","docker","kubernetes",
                         "react","java","c++","nlp","machine learning"]

        for r in resumes[:10]:
            try:
                content = await r.read()
                text = extract_text_generic(io.BytesIO(content), r.filename)
                if len(text.strip()) == 0:
                    print(f"Warning: No text extracted from {r.filename}")

        # Scores
        tf_score = simple_tfidf_score(jd_text, text)
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
            print(f"Skill extraction failed for {r.filename}: {e}")
            found, missing = [], []

        # Remarks
        if final_score < 40:
            remarks = "Not a strong match"
        elif final_score < 70:
            remarks = "Partial match"
        else:
            remarks = "Strong match"

        results.append({
            "filename": r.filename,
            "score": round(float(final_score), 2),
            "found_skills": found,
            "missing_skills": missing,
            "remarks": remarks
        })

    except Exception as e:
        print(f"Error processing {r.filename}: {e}")
        results.append({
            "filename": r.filename,
            "score": 0,
            "found_skills": [],
            "missing_skills": [],
            "remarks": "Error processing file"
        })

# ----------------------
# Generate Emails
# ----------------------
@app.post("/generate_emails")
async def generate_emails(candidate_name: str = Form(...),
                          candidate_email: str = Form(...),
                          decision: str = Form("interview"),
                          role: str = Form("")):
    if not client:
        return JSONResponse({"error": "OPENAI_API_KEY not set."})

    if decision.lower() == "interview":
        prompt = f"Write a professional interview call email to {candidate_name} for the role {role}. Provide subject and body."
    else:
        prompt = f"Write a polite rejection email to {candidate_name} for the role {role}. Provide subject and body."

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        email_content = resp.choices[0].message.content
        return JSONResponse({"email": email_content})
    except Exception as e:
        print("Email generation error:", e)
        return JSONResponse({"error": f"Failed to generate email: {str(e)}"})
