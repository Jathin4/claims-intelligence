from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import cases as case_runner
from app.routers import auth, cases, claims, models


@asynccontextmanager
async def lifespan(app: FastAPI):
    case_runner.resume_unfinished()      # investigations a previous process left unfinished
    yield


app = FastAPI(title="AI Claims Intelligence Platform", lifespan=lifespan,
              description="Review-priority scores for claims. Synthetic data; a score is a signal for review, never proof of fraud.")
app.include_router(auth.router)
app.include_router(claims.router)
app.include_router(models.router)
app.include_router(cases.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
