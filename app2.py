import uvicorn
import pandas as pd
from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from src.pipeline.predict_pipeline import FormData, PredictPipeline


app = FastAPI()

# Setting up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Route for a home page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/predictdata", response_class=HTMLResponse)
async def predict_datapoint(request: Request, data: FormData = Depends()):
    # Convert the FormData object into a dictionary suitable for prediction
    data = data.dict()
    prediction_dataframe = pd.DataFrame(data)

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(prediction_dataframe)

    if results[0] > 0.5:
        results = "Not Purchase"
    else:
        results = "Purchase"

    return templates.TemplateResponse("index.html", {"request": request, "results": results})


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
