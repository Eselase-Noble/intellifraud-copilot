from pydantic import BaseModel, Field

class InvestigationRequest(BaseModel):
    query: str = Field(..., description="Analyst question or investigation request")

class InvestigationResponse(BaseModel):
    result: dict

class ReindexRequest(BaseModel):
    force: bool = Field(False, description="Force rebuild index even if exists")

class ReindexResponse(BaseModel):
    status: str
    chunks_indexed: int
