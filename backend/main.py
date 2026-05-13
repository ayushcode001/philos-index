from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from fastapi.middleware.cors import CORSMiddleware

import os
import spacy
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb


from features import get_sdi, get_scd, get_ar, get_ld, get_vrs, get_mattr, get_parse_tree_depth, get_philosophical_compound_density, get_abstract_noun_repetition, get_pvd, get_slv

app = FastAPI(title='Philos API', description='AI Philosophical Density Analyzer')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # will be resticting this with vue.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")
print(f"Loading NLP model ({SPACY_MODEL}) and ML Models into memory")

try:
    nlp = spacy.load(SPACY_MODEL)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
    model_path = os.path.join(BASE_DIR, "models", "xgb_model.pkl")

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    print(" Models loaded successfully.")

except Exception as e:
    print(f"Error loading models: {e}")


# Define the data contract (what the frontend must send)
class TextRequest(BaseModel):
    text: str

#Check Route
@app.get("/")
def read_root():
    return {"status": "Philomath Engine is Active."}

# analyse endpoint
@app.post("/analyze")
async def analyze_text(request: TextRequest):
    """Takes raw text, extracts features, and returns the Philomath Score."""
    raw_text = request.text.strip()
    
    if len(raw_text.split()) < 30:
        raise HTTPException(status_code=400, detail="Text too short. Provide at least 30 words.")

    try:
        # Parse the text
        doc = nlp(raw_text)
        
        # Extract the 11 linguistic features
        features = [
            get_sdi(doc),
            get_scd(doc),
            get_ar(doc),
            get_ld(doc),
            get_vrs(doc),
            get_mattr(doc),
            get_parse_tree_depth(doc),
            get_philosophical_compound_density(doc),
            get_abstract_noun_repetition(doc),
            get_pvd(doc),
            get_slv(doc)
            
        ]
        
        # Convert to numpy array and shape for the model
        # feature_array = np.array(features).reshape(1, -1)
        feature_df = pd.DataFrame([features], columns=['SDI', 'SCD', 'AR', 'LD', 'VRS', 'MATTR', 'PTD', 'PCD', 'ANR', 'PVD', 'SLV'])

        # Scale the data using the exact boundaries from training
        # scaled_features = scaler.transform(feature_array)
        scaled_features = scaler.transform(feature_df)
        
        # Predict the score
        raw_prediction = model.predict(scaled_features)[0]
        
        # Clamp the score between 0 and 100 for safety
        final_score = max(0.0, min(100.0, round(float(raw_prediction), 1)))
        
        # Return the payload to the frontend
        return {
            "score": final_score,
            "features_extracted": {
                "SDI": round(features[0], 2),
                "SCD": round(features[1], 2),
                "AR": round(features[2], 2),
                "LD": round(features[3], 2),
                "VRS": round(features[4], 2),
                "MATTR": round(features[5], 2),
                "PTD": round(features[6], 2),
                "PCD": round(features[7], 2),
                "ANR": round(features[8], 2),
                "PVD": round(features[9], 2),
                "SLV": round(features[10], 2)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis Failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)