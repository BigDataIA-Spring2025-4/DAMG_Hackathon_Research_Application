import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.hospital_trends.integrated import generate_integrated_report

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

class NVDIARequest(BaseModel):
    state: str
    
app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

@app.get("/")
def read_root():
    return {"message": "Agentic Research Tool"}


@app.post("/generate_research")
def query_nvdia_documents(request: NVDIARequest):
    try:
        state = request.state
        print("state:", state)

        report = generate_integrated_report(state)
        print("report generated")
        answer = report
        
        return {
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")