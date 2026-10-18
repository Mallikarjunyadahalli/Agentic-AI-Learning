
from pydantic import BaseModel, AnyUrl, Field
from typing import List, Dict, Optional, Annotated

class PatientData(BaseModel):
    name:Annotated[str, Field(max_length=10, title="Name of the patient", description="This is the name of the patient", Example="Mallu")] = ""
    age: Annotated[int, Field(gt=0, lt=50)] = 0
    email: Annotated[str, Field(max_length=100)]
    linkedIn: Annotated[AnyUrl, Field()]
    weight: Annotated[float, Field(gt=0, description="Weight of the patient in kg", strict=True)]
    married: Annotated[bool, Field(default=False)]
    allergies: Annotated[Optional[List[str]], Field(max_length=5)] = None
    contactInfo: Annotated[Dict[str, str], Field()]

def add_patient_data(patient: PatientData):
    print(f"Name: {patient.name}")
    print(f"Age: {patient.age}")
    print(f"Email: {patient.email}")
    print(f"Linked In: {patient.linkedIn}")
    print(f"Weight: {patient.weight}")
    print(f"Married: {patient.married}")
    print(f"Allergies: {patient.allergies}")
    print(f"Contact information: {patient.contact_info}")

name = input("Name: ")
age = int(input("Age: "))
email = input("Email: ")
linkedIn = input("Linked In: ")
weight = float(input("Weight: "))
married = bool(input("Married: "))
allergies = input("Married (seperated by ',') : ").split(",")
contact_info = int(input("Contact number: "))


patieent_info = {"name" : name, "age" : age, "email" : email, "weight" : weight, "married" : married, "contact_info" : contact_info}

patient_1 = add_patient_data(**patieent_info)
print(patient_1)
