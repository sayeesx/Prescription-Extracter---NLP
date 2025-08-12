"""
Web application using FastAPI for document processing.
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    """Process an uploaded document."""
    pass

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Document Processing API"}
