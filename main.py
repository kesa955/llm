from typing import Union
from fastapi import FastAPI, Query
from transformers import pipeline

# ONE FastAPI app
app = FastAPI(
    title="My LLM API",
    description="This API manages items using FastAPI + Swagger.",
    version="1.0.0",
    docs_url="/docs",           # Swagger UI
    redoc_url="/redoc"          # ReDoc docs
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/items/")
def read_items(q: Union[str, None] = Query(default=None, description="Search query")):
    return {"q": q}

# Load a small model for testing
llm = pipeline("text-generation", model="gpt2")  # or try "tiiuae/falcon-rw-1b", "mistralai/Mistral-7B-Instruct-v0.1" etc.

@app.post("/generate/")
def generate_text(prompt: str):
    response = llm(prompt, max_length=100, do_sample=True)
    return {"response": response[0]['generated_text']}