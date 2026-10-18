from pydantic import BaseModel, AnyUrl, field_validator
from typing import List, Dict, Optional

class PatientData(BaseModel):
    name: str
    age: int
    email: str
    weight: float
    maried: bool = False  #we can add default value also
    allergies: Optional[List[str]] = None
    contact_info: Dict[str, str]  ## Key and value are string

    @field_validator('email')
    @classmethod
    def email_validator(cls, value):
        valid_domains = ["hdfc.com", 'icici.com']
        domain_name = value.split("@")[-1]

        if domain_name not in valid_domains:
            raise ValueError("Not a valid domain")

        return value

def add_patient_data(patient: PatientData):
    print(f"Name: {patient.name}")
    print(f"Age: {patient.age}")
    print(f"Email: {patient.email}")
    print(f"Weight: {patient.weight}")
    print(f"Married: {patient.maried}")
    print(f"Allergies: {patient.allergies}")
    print(f"Contact information: {patient.contact_info}")


patient_info = {"name" : "Mallu", "age" : 20, "email" : "mallu@icici.com", "weight" : 70.5, "maried" : False, "allergies" : ["peanuts"], "contact_info" : {"phone" : "123-456-7890", "address" : "123 Main St"}}

patient_1 = PatientData(**patient_info)
add_patient_data(patient_1)