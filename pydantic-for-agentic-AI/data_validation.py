
from pydantic import BaseModel, EmailStr, AnyUrl
from typing import List, Dict, Optional

class PatientData(BaseModel):
    name: str
    age: int
    email: EmailStr
    linked_in: AnyUrl
    weight: float
    maried: bool = False  #we can add default value also
    allergies: Optional[List[str]] = None
    contact_info: Dict[str, str]  ## Key and value are string

def add_patient_data(patient: PatientData):
    print(f"Name: {patient.name}")
    print(f"Age: {patient.age}")
    print(f"Email: {patient.email}")
    print(f"Weight: {patient.weight}")
    print(f"Married: {patient.maried}")
    print(f"Allergies: {patient.allergies}")
    print(f"Contact information: {patient.contact_info}")


def update_patient_info(patient: PatientData):
    print(f"Name: {patient.name}")
    print(f"Age: {patient.age}")
    print(f"Weight: {patient.weight}")
    print(f"Married: {patient.maried}")
    print(f"Allergies: {patient.allergies}")
    print(f"Contact information: {patient.contact_info}")


patient_info = {"name" : "Mallu", "age" : 20, "email" : "hi@gmail.com" ,"weight" : 5.1, "maried" : False, "contact_info" : {"email" : "mallu@9090", "phone_number" : '024894515421'}}

patient_1 = PatientData(**patient_info)

add_patient_data(patient_1)