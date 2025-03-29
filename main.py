from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import io
import json
from PIL import Image

# LangChain and OpenAI imports:
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Import Hugging Face Transformers for image captioning
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


app = FastAPI(title="Donation Categorization API")

# CORS configuration: allowing requests from any origin (you can change it for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Allow all origins for testing; change this for production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Example predetermined list of donation categories (adjust as needed)
CATEGORIES = ["Food", "Clothing", "Toys", "Books", "Electronics", "Household", "Furniture"]

# Create an OpenAI LLM instance (make sure to set your API key in your environment)
llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4-turbo",  
    streaming=False,  # synchronous call for our API endpoint
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Define a prompt template for donation categorization
prompt_template = PromptTemplate(
    input_variables=["donation_text", "categories"],
    template="""
You are an assistant that categorizes donation items. Given the donation description:
{donation_text}

From the following list of categories: {categories},
determine the best matching category for this donation.

Return your answer in **strict JSON** format with no additional text:
- If a valid category is found, output exactly: {{"category": ["Category Name"]}}.
- If no appropriate category is found, output: {{"status": "NoCategory"}}.
- If the donation contains inappropriate content or trolling, output: {{"status": "TrollDetected"}}.

Only output valid JSON.
"""
)


class DonationResponse(BaseModel):
    category: list = None
    status: str = None

# Initialize the image captioning model and processor
model_name = "nlpconnect/vit-gpt2-image-captioning"
caption_model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_caption(image: Image.Image) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    # You can adjust generation parameters as needed
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

    # Prefer text input over file upload if both are provided.
    if text:
        donation_text = text.strip()
    elif file:
        # If the file is an image, use image captioning to extract a description.
        if file.content_type.startswith("image/"):
            try:
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))
                donation_text = generate_caption(image).strip()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")
        else:
            # For non-image files, assume text-based content.
            try:
                file_bytes = await file.read()
                donation_text = file_bytes.decode("utf-8").strip()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="No input provided. Please send a text description or upload a file.")

    if not donation_text:
        raise HTTPException(status_code=400, detail="No text could be extracted from the input.")

    # Format the prompt with the donation text and category list.
    prompt = prompt_template.format(
        donation_text=donation_text,
        categories=", ".join(CATEGORIES)
    )
    print(prompt)

    # Get a response from the LLM.
    try:
        llm_response = llm.call_as_llm(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM processing error: {str(e)}")

    # Try to parse the LLM's output as JSON.
    try:
        response_json = json.loads(llm_response)
        # print(donation_text+":",llm_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM response as JSON: {str(e)}")

    return response_json

# Do not run the server for deployment; 
# uncomment the following lines for local testing only.
# if __name__ == "__main__":
#     # Run the API with uvicorn on port 8000
#     uvicorn.run(app, host="0.0.0.0", port=8000)
