from diffusers import StableDiffusionPipeline
from fastapi import FastAPI, HTTPException
import torch
import base64
from io import BytesIO
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="LLM Text Generation API",
    description="Generate text using Hugging Face distilgpt2 model.",
    version="1.0"
)

# Load image generation pipeline
image_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
).to("cuda" if torch.cuda.is_available() else "cpu")

# Request schema for image generation
class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate-image")
def generate_image(request: ImageRequest):
    try:
        image = image_pipe(request.prompt).images[0]

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        #commented lines to convert image to base64
        #img_str = base64.b64encode(buffered.getvalue()).decode()
        #return JSONResponse(content={"image_base64": img_str})
        return StreamingResponse(buffer, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#below logic to get png from base64
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# import io
# import base64

# @app.get("/image")
# def get_image():
#     # Example base64 string (replace with your actual base64 string)
#     image_base64 = "iVBO........"

#     # Decode the base64 string to bytes
#     image_bytes = base64.b64decode(image_base64)

#     # Wrap the bytes in a BytesIO object so it acts like a file
#     image_stream = io.BytesIO(image_bytes)

#     # Return the image as a StreamingResponse with the correct media type
#     return StreamingResponse(image_stream, media_type="image/png")
