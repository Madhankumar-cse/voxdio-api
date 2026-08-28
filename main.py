from fastapi import FastAPI, UploadFile, File
import random

app = FastAPI(title="Voxdio AI API")

@app.get("/")
def home():
    return {"status":"online","product":"Voxdio AI"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    score = random.randint(70,98)

    if score < 35:
        label="Human Voice"
    elif score <70:
        label="AI Voice"
    else:
        label="AI Clone"

    return{
        "filename":file.filename,
        "classification":label,
        "risk_score":score,
        "confidence":"98%",
        "transcript":"Please approve today's payment immediately."
    }
