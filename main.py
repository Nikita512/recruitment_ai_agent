# main.py
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from utils.text_extract import extract_text_generic
from utils.scoring import embedding_score, extract_skills_from_text, simple_tfidf_score
import os, io, uuid
from openai import OpenAI
from typing import List
from dotenv import load_dotenv

# Load .env file for local testing
load_dotenv()

# --- Setup ---
os.makedirs("jds", exist_ok=True)
os.makedirs("resumes", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app = FastAPI(title="Recruitment AI Agent")
templates = Jinja2Templates(directory="templates")

# --- OpenAI client setup ---
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serve frontend"""
    return templates.TemplateResponse("index.html", {"request": request})


# --- Upload JD (either paste or file) ---
@app.post("/upload_jd")
async def upload_jd(jd_text: str = Form(None), jd_file: UploadFile = File(None)):
    try:
        text = ""
        if jd_file is not None:
            content = await jd_file.read()
            text = extract_text_generic(io.BytesIO(content), jd_file.filename)
        elif jd_text and jd_text.strip():
            text = jd_text.strip()
        else:
            return JSONResponse({"error": "Please paste JD text or upload a JD file."}, status_code=400)

        jid = f"jd_{uuid.uuid4().hex[:8]}.txt"
        with open(os.path.join("jds", jid), "w", encoding="utf-8") as f:
            f.write(text)

        return {"status": "saved", "jd_file": jid, "message": "JD saved successfully!"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# --- Upload resumes and compute scores ---
@app.post("/upload_resumes")
async def upload_resumes(resumes: List[UploadFile] = File(...)):
    try:
        jd_files = sorted(os.listdir("jds"))
        if not jd_files:
            return {"error": "No JD found. Please upload or paste a JD first."}

        with open(os.path.join("jds", jd_files[-1]), "r", encoding="utf-8") as f:
            jd_text = f.read()

        common_skills = [
            "python", "fastapi", "sql", "aws", "docker", "kubernetes",
            "react", "java", "c++", "nlp", "machine learning"
        ]

        results = []
        for resume in resumes[:10]:
            content = await resume.read()
            text = extract_text_generic(io.BytesIO(content), resume.filename)

            emb_score = 0.0
            tf_score = 0.0

            try:
                emb_score = embedding_score(jd_text, text)
            except Exception as e:
                print("Embedding scoring error:", e)

            try:
                tf_score = simple_tfidf_score(jd_text, text)
            except Exception as e:
                print("TF-IDF scoring error:", e)

            final_score = (emb_score * 0.7) + (tf_score * 0.3)
            found, missing = extract_skills_from_text(text, common_skills)

            if final_score < 40:
                remark = "Not a strong match"
            elif final_score < 70:
                remark = "Partial match"
            else:
                remark = "Strong match"

            results.append({
                "filename": resume.filename,
                "score": round(float(final_score), 2),
                "found_skills": found,
                "missing_skills": missing,
                "remarks": remark
            })

        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        return {"results": results_sorted}

    except Exception as e:
        print("Upload/score error:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


# --- Generate email ---
@app.post("/generate_emails")
async def generate_emails(
    candidate_name: str = Form(...),
    candidate_email: str = Form(...),
    decision: str = Form("interview"),
    role: str = Form("")
):
    try:
        if not client:
            return {"error": "OpenAI API key not configured."}

        if decision == "interview":
            prompt = f"Write a professional interview invitation email to {candidate_name} for the {role} role. Include subject and body."
        else:
            prompt = f"Write a polite rejection email to {candidate_name} for the {role} role. Include subject and body."

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return {"email": response.choices[0].message.content}

    except Exception as e:
        error_str = str(e)
        print("Email generation error:", error_str)
        if "insufficient_quota" in error_str:
            return {
                "email": f"Dear {candidate_name},\n\n(This is a demo message: your OpenAI quota is exceeded.)\n\nThank you for applying for the {role} role.\n\nBest,\nRecruitment Team"
            }
        return {"error": f"Failed to generate email: {error_str}"}
