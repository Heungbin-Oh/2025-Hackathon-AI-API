from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import io
import json
from PIL import Image
from openai import OpenAI

# Transformers for image captioning
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Initialize FastAPI app
app = FastAPI(title="Donation Categorization API")

# Allow all CORS (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Categories to classify
CATEGORIES = ["Food", "Clothing", "Toys", "Books", "Electronics", "Household", "Furniture"]

# OpenAI client (new SDK)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Prompt template
prompt_template = '''
You are an assistant that categorizes donation items. Given the donation description:
{donation_text}

From the following list of categories: {categories},
determine the best matching category for this donation.

Return your answer in **strict JSON** format with no additional text:
- If a valid category is found, output exactly: {"category": ["Category Name"]}.
- If no appropriate category is found, output: {"status": "NoCategory"}.
- If the donation contains inappropriate content or trolling, output: {"status": "TrollDetected"}.

Only output valid JSON.
'''

class DonationResponse(BaseModel):
    category: list = None
    status: str = None

# Image captioning setup
model_name = "nlpconnect/vit-gpt2-image-captioning"
caption_model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_caption(image: Image.Image) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    output_ids = caption_model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

@app.get("/")
async def root():
    return {"message": "Welcome to the Donation Categorization API!"}

@app.post("/categorize_donation", response_model=dict)
async def categorize_donation(
    text: str = Form(None),
    file: UploadFile = File(None)
):
    donation_text = ""

    if text:
        donation_text = text.strip()
    elif file:
        if file.content_type.startswith("image/"):
            try:
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))
                donation_text = generate_caption(image).strip()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")
        else:
            try:
                file_bytes = await file.read()
                donation_text = file_bytes.decode("utf-8").strip()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="No input provided.")

    if not donation_text:
        raise HTTPException(status_code=400, detail="No text could be extracted from the input.")

    prompt = prompt_template.format(
        donation_text=donation_text,
        categories=", ".join(CATEGORIES)
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600
        )
        result = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM processing error: {str(e)}")

    try:
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM response as JSON: {str(e)}")
