from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
from .decision_engine import final_decision

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(
    leaf: UploadFile = File(...),
    pest: UploadFile = File(...)
):

    leaf_path = "temp_leaf.jpg"
    pest_path = "temp_pest.jpg"

    with open(leaf_path, "wb") as buffer:
        shutil.copyfileobj(leaf.file, buffer)

    with open(pest_path, "wb") as buffer:
        shutil.copyfileobj(pest.file, buffer)

    soil_values = [90,42,43,20.5,82,6.5,202]

    result = final_decision(
        soil_values,
        leaf_path,
        pest_path
    )

    return result