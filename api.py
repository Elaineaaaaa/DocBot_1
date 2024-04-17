from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import os

app = FastAPI()

class Item(BaseModel):
    query: str
    model_name: str

@app.post("/process-files/")
async def process_files(file: UploadFile = File(...)):
    # Process the file similar to how it's done in Streamlit
    return {"filename": file.filename}

@app.post("/ask-question/")
async def ask_question(item: Item):
    # Placeholder for your question-answering logic
    response = f"Answer to your question '{item.query}' using model '{item.model_name}'"
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
