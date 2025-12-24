from fastapi import APIRouter, HTTPException
from app.api.schemas import InvestigationRequest, InvestigationResponse, ReindexRequest, ReindexResponse
from app.agents.planner import PlannerAgent
from app.rag.retriever import PolicyIndex
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/investigate", response_model=InvestigationResponse)
async def investigate(req: InvestigationRequest):
    try:
        planner = PlannerAgent()
        result = await planner.run(req.query)
        return InvestigationResponse(result=result)
    except Exception as e:
        logger.exception("investigate failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/reindex", response_model=ReindexResponse)
async def rag_reindex(req: ReindexRequest):
    idx = PolicyIndex()
    chunks = idx.build_or_load(force=req.force)
    return ReindexResponse(status="ok", chunks_indexed=len(chunks))
