import os
import shutil
import requests
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from main import run_diagnosis  # Import your AI logic

app = FastAPI()

# Configuration
DATA_FOLDER = "data"
VERCEL_API_URL = "https://your-vercel-project.vercel.app/api/receive-result"
os.makedirs(DATA_FOLDER, exist_ok=True)

@app.post("/upload-and-analyze")
async def upload_and_analyze(file: UploadFile = File(...)):
    try:
        # 1. Save the received file to the 'data' folder
        file_path = os.path.join(DATA_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Run the AI Analysis from main.py
        # We pass the path of the saved file
        analysis_result = run_diagnosis(file_path)

        # 3. Add metadata to the result (optional)
        payload = {
            "filename": file.filename,
            "result": analysis_result
        }

        # 4. Send the result back to your Vercel server
        vercel_response = requests.post(
            VERCEL_API_URL, 
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        # 5. Cleanup: Delete the file after processing to save space
        os.remove(file_path)

        return {
            "message": "Analysis complete and sent to Vercel",
            "vercel_status": vercel_response.status_code,
            "analysis": analysis_result
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # EC2 usually uses port 80 or 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
