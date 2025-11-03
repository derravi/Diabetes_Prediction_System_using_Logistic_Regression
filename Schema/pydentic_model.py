from pydantic import BaseModel,Field
from typing import Annotated

class UserInput(BaseModel):

    Pregnancies:Annotated[int,Field(...,gt=0,description="Enter the Pregnancies.",examples=["Like 8"])]
    Glucose:Annotated[int,Field(...,gt=0,description="Enter the Glucose.",examples=["Like 130"])]
    BloodPressure:Annotated[int,Field(...,gt=0,description="Enter the BloodPressure.",examples=["Like 50 to 90"])]
    SkinThickness:Annotated[int,Field(...,gt=0,description="Enter the SkinThickness",examples=["20 to 40"])]
    Insulin:Annotated[int,Field(...,gt=0,description="Enter the Insuli",examples=['Like 0 to 200'])]
    BMI:Annotated[int,Field(...,gt=0,description="Enter the BMI",examples=["Like 20 to 60"])]
    DiabetesPedigreeFunction:Annotated[float,Field(...,gt=0,description="Enter the DiabetesPedigreeFunction.",examples=["Like 0.00 to 5"])]
    Age:Annotated[int,Field(...,description="Enter the Age.",examples=[50])]
