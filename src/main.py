import shutil
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from src.inference.predictor import NeuroVoxPredictor

app = FastAPI()

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "checkpoint" / "best_model.onnx"
UPLOAD_DIR = BASE_DIR / "temp_uploads"

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Global variable to hold the model
predictor = None

@app.on_event("startup")
async def load_model():
    """Load the model once when the server starts to save time."""
    global predictor
    print(f"Loading model from: {MODEL_PATH}")
    try:
        predictor = NeuroVoxPredictor(str(MODEL_PATH))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model. {e}")

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Endpoint that receives an audio file, saves it, and runs inference.
    """
    if not predictor:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # 1. Save the uploaded file temporarily
    temp_file_path = UPLOAD_DIR / file.filename
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Run your existing inference logic
        label, prob = predictor.predict(temp_file_path)
        
        # 3. Clean up (delete the temp file)
        os.remove(temp_file_path)

        # 4. Return results as JSON
        return {
            "diagnosis": label,
            "confidence": float(prob), # Convert numpy float to python float
            "filename": file.filename
        }

    except Exception as e:
        return {"error": str(e)}

# Serve the frontend HTML file directly
@app.get("/")
async def get_client():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())
