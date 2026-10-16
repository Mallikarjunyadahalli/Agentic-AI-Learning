
from pydantic import BaseModel

class PatientData(BaseModel):
    name: str
    age: int

def add_patient_data(patient: PatientData):
    print(f"Name: {patient.name}")
    print(f"Age: {patient.age}")

    print("Database Addedd Sucessfully.")

def update_patient_data(patient: PatientData):
    print(f"Name: {patient.name}")
    print(f"Age: {patient.age}")

    print("Database Updated Sucessfully.")


patient_data = {"name" : "Mallu", "age" : 20}

patient = PatientData(**patient_data)

add_patient_data(patient)
update_patient_data(patient)