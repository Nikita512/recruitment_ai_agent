# Recruitment AI Agent (FastAPI)

This project implements a Recruitment AI Agent as specified in the provided task document. Features:
- Upload or paste Job Descriptions (JD)
- Generate JD from structured inputs using LLM
- Upload up to 10 resumes (PDF/DOCX)
- Extract text and compute match score vs JD
- Generate interview / rejection emails with an LLM

## How to run in Colab
1. Add your OpenAI API key in `COLAB` when prompted or set an environment variable `OPENAI_API_KEY`.
2. Run the Colab cells that install dependencies, create files, and then run the FastAPI app; the notebook will expose the local app via ngrok.

## Model choice
- Uses OpenAI GPT (text + embeddings) by default for JD generation and email; for scoring we include a fallback TF-IDF + sentence-transformer similarity approach for cost control.

See code files in the repository.