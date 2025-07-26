from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class PostContent(BaseModel):
    narrative: str
    characters: List[str]
    inventory: List[str]
    previous_image_url: str
    previous_image_style: str

@app.post("/generate")
async def generate(content: PostContent):
    # Şu an sadece gelen veriyi döndürüyoruz
    return {"message": "Post received successfully", "data": content}