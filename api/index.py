from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import io
import base64
import json
import openai

app = FastAPI(title="Donation Categorization API (Lite)")

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CATEGORIES = ["Food", "Clothing", "Toys", "Books", "Electronics", "Household", "Furniture"]
openai.api_key = os.getenv("OPENAI_API_KEY")

class DonationResponse(BaseModel):
    category: list | None = None
    status: str | None = None

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI on Vercel!"}

@app.post("/categorize_donation", response_model=dict)
async def categorize_donation(
    text: str = Form(None),
    file: UploadFile = File(None)
):
    if not text and not file:
        raise HTTPException(status_code=400, detail="Send a text description or image file")

    if not text and file:
        # Convert image to base64
        img_bytes = await file.read()
        b64_image = base64.b64encode(img_bytes).decode("utf-8")

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the object in this donation image."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                        ]
                    }
                ],
                max_tokens=1000
            )
            text = response.choices[0].message.content.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image captioning failed: {str(e)}")

    prompt = f"""
    You are an assistant categorizing donations. Given this item:
    "{text}"

    Choose the best category from this list: {', '.join(CATEGORIES)}.

    Respond with strict JSON only:
    - {"category": ["Category Name"]} if matched
    - {"status": "NoCategory"} if none
    - {"status": "TrollDetected"} if inappropriate
    """

    try:
        result = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = result.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Categorization error: {str(e)}")
