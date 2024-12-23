from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run
import sys

from typing import Optional, Dict

from us_visa_prediction.constants import APP_HOST, APP_PORT
from us_visa_prediction.pipline.prediction_pipeline import USvisaData, USvisaClassifier
from us_visa_prediction.pipline.training_pipeline import TrainPipeline

# Model configuration
MODEL_CONFIG = {
    "model_container_name": "usvisamlops",
    "model_file_path": "model.pkl"  # Updated path to include models directory
}

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.continent: Optional[str] = None
        self.education_of_employee: Optional[str] = None
        self.has_job_experience: Optional[str] = None
        self.requires_job_training: Optional[str] = None
        self.no_of_employees: Optional[str] = None
        self.company_age: Optional[str] = None
        self.region_of_employment: Optional[str] = None
        self.prevailing_wage: Optional[str] = None
        self.unit_of_wage: Optional[str] = None
        self.full_time_position: Optional[str] = None

    async def get_usvisa_data(self):
        try:
            form = await self.request.form()
            self.continent = form.get("continent")
            self.education_of_employee = form.get("education_of_employee")
            self.has_job_experience = form.get("has_job_experience")
            self.requires_job_training = form.get("requires_job_training")
            self.no_of_employees = form.get("no_of_employees")
            self.company_age = form.get("company_age")
            self.region_of_employment = form.get("region_of_employment")
            self.prevailing_wage = form.get("prevailing_wage")
            self.unit_of_wage = form.get("unit_of_wage")
            self.full_time_position = form.get("full_time_position")
        except Exception as e:
            raise Exception(f"Error in getting form data: {str(e)}")

@app.get("/", tags=["authentication"])
async def index(request: Request):
    try:
        return templates.TemplateResponse(
                "usvisa.html",
                {"request": request, "context": "Rendering"}
        )
    except Exception as e:
        return Response(f"Error occurred! {e}", status_code=500)

@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/")
async def predictRouteClient(request: Request):
    try:
        # Get form data
        form = DataForm(request)
        await form.get_usvisa_data()
        
        # Create usvisa data instance
        usvisa_data = USvisaData(
            continent=form.continent,
            education_of_employee=form.education_of_employee,
            has_job_experience=form.has_job_experience,
            requires_job_training=form.requires_job_training,
            no_of_employees=form.no_of_employees,
            company_age=form.company_age,
            region_of_employment=form.region_of_employment,
            prevailing_wage=form.prevailing_wage,
            unit_of_wage=form.unit_of_wage,
            full_time_position=form.full_time_position
        )
        
        # Get DataFrame
        usvisa_df = usvisa_data.get_usvisa_input_data_frame()
        
        # Initialize classifier with model configuration
        model_predictor = USvisaClassifier(prediction_pipeline_config=MODEL_CONFIG)
        
        # Check if model exists before prediction
        if not model_predictor.estimator.is_model_present():
            return templates.TemplateResponse(
                "usvisa.html",
                {
                    "request": request, 
                    "context": "Error: Model not found in storage. Please ensure the model is trained and uploaded."
                },
                status_code=404
            )
        
        # Make prediction
        value = model_predictor.predict(dataframe=usvisa_df)[0]
        status = "Visa-approved" if value == 1 else "Visa Not-Approved"
        
        return templates.TemplateResponse(
            "usvisa.html",
            {"request": request, "context": status}
        )
        
    except Exception as e:
        error_message = f"Prediction failed: {str(e)}"
        return templates.TemplateResponse(
            "usvisa.html",
            {"request": request, "context": error_message},
            status_code=500
        )

if __name__ == "__main__":
    try:
        app_run(app, host=APP_HOST, port=APP_PORT)
    except Exception as e:
        print(f"Error occurred while starting the application: {str(e)}")
        sys.exit(1)