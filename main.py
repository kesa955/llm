# from typing import Union
# from fastapi import FastAPI, Query
# from transformers import pipeline

# # ONE FastAPI app
# app = FastAPI(
#     title="My LLM API",
#     description="This API manages items using FastAPI + Swagger.",
#     version="1.0.0",
#     docs_url="/docs",           # Swagger UI
#     redoc_url="/redoc"          # ReDoc docs
# )

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

# @app.get("/items/")
# def read_items(q: Union[str, None] = Query(default=None, description="Search query")):
#     return {"q": q}

# # Load a small model for testing
# llm = pipeline("text-generation", model="gpt2")  # or try "tiiuae/falcon-rw-1b", "mistralai/Mistral-7B-Instruct-v0.1" etc.

# @app.post("/generate/")
# def generate_text(prompt: str):
#     response = llm(prompt, max_length=100, do_sample=True)
#     return {"response": response[0]['generated_text']}

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI(
    title="LLM Text Generation API",
    description="Generate text using Hugging Face distilgpt2 model.",
    version="1.0"
)

# ✅ This model works with HF Inference API
API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom-560m"
headers = {
    "Authorization": "Bearer your_token"  # ← Replace with your real token
}

class Prompt(BaseModel):
    text: str

@app.post("/generate/")
def generate_text(prompt: Prompt):
    payload = {"inputs": prompt.text}
    
    # ✅ Debug: print payload and URL
    print(f"\n[DEBUG] Sending request to Hugging Face API")
    print(f"[DEBUG] URL: {API_URL}")
    print(f"[DEBUG] Headers: {headers}")
    print(f"[DEBUG] Payload: {payload}")
    
    response = requests.post(API_URL, headers=headers, json=payload)

    # ✅ Debug: print status code and raw response
    print(f"[DEBUG] Response status code: {response.status_code}")
    print(f"[DEBUG] Raw response text: {response.text}\n")

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"Hugging Face API Error: {response.text}")

    try:
        output = response.json()
        print(f"[DEBUG] Parsed JSON response: {output}")
        return {
            "response": output[0]["generated_text"] if "generated_text" in output[0] else output[0]
        }
    except Exception as e:
        print(f"[ERROR] Exception while parsing response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing response: {str(e)}")