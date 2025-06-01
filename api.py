from transformers import pipeline

# Load a small model for testing
llm = pipeline("text-generation", model="gpt2")  # or try "tiiuae/falcon-rw-1b", "mistralai/Mistral-7B-Instruct-v0.1" etc.

@app.post("/generate/")
def generate_text(prompt: str):
    response = llm(prompt, max_length=100, do_sample=True)
    return {"response": response[0]['generated_text']}