# state.py

from typing import TypedDict, Annotated, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
import operator

class Medication(BaseModel):
    id: Optional[int] = Field(None, description="Unique identifier for the drug")
    drug_name: Optional[str] = Field(None, description="Brand name of the drug")
    generic_name: Optional[str] = Field(None, description="Generic name of the drug")
    drug_class: Optional[str] = Field(None, description="Class or category of the drug")
    indications: Optional[str] = Field(None, description="Approved uses or indications for the drug")
    dosage_form: Optional[str] = Field(None, description="Physical form of the drug (e.g., tablet, capsule, injection)")
    strength: Optional[str] = Field(None, description="Strength or concentration of the active ingredient")
    route_of_administration: Optional[str] = Field(None, description="How the drug is administered (e.g., oral, intravenous)")
    mechanism_of_action: Optional[str] = Field(None, description="How the drug works or its mode of action")
    side_effects: Optional[str] = Field(None, description="Potential side effects of the drug")
    contraindications: Optional[str] = Field(None, description="Situations or conditions where the drug should not be used")
    interactions: Optional[str] = Field(None, description="Potential interactions with other drugs or substances")
    warnings_and_precautions: Optional[str] = Field(None, description="Important warnings and precautions for using the drug")
    pregnancy_category: Optional[str] = Field(None, description="Category indicating the potential risk during pregnancy")
    storage_conditions: Optional[str] = Field(None, description="Recommended storage conditions for the drug")
    manufacturer: Optional[str] = Field(None, description="Company that manufactures the drug")
    approval_date: Optional[str] = Field(None, description="Date the drug was approved for use")
    availability: Optional[str] = Field(None, description="Information about the availability of the drug")
    ndc: Optional[str] = Field(None, description="National Drug Code (NDC) for the drug")
    price: Optional[float] = Field(None, description="Price or cost of the drug")

class MedicationDictionary(BaseModel):
    medications: Dict[str, Medication] = {}

class MedicationList(BaseModel):
    medications: List[Medication] = []

# State Initialization
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    medication_info: list[Medication]
    prescription_details: str
    sender: str
    final_reply: str
    nearby_pharmacies: str
    location: str


class Medicamento_Protegido(BaseModel):
    principio_activo: str = Field(alias="Principio activo")
    registro_sanitario: str = Field(alias="Registro sanitario")
    titular_registro_sanitario: str = Field(alias="Titular Reg. Sanitario")
    especialidad_farmaceutica: str = Field(alias="Especialidad farmacéutica")
    forma_farmaceutica: str = Field(alias="Forma farmacéutica")
    dosis: str = Field(alias="Dosis")
    presentacion_por_envase: str = Field(alias="Presentación x envase")
    estupefaciente_o_psicotropico: bool = Field(alias="Estupefaciente o Psicotrópico")

# Definir el esquema de salida para el agente de farmacia
class PharmacyResponse(BaseModel):
    pharmacy_name: str
    address: str
    city: str
    region: str
    opening_hours: str
    closing_hours: str
    distance: str