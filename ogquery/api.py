from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from pathlib import Path
import tempfile
import shutil

from ogquery.core import OGQuery

class QueryRequest(BaseModel):
    dataset_id: str
    query: str


def create_app(engine: "OGQuery") -> FastAPI:
    """
    Factory pattern:
    API layer is fully dependent on OGQuery instance.
    """

    app = FastAPI(title="OGQuery API", version="1.0.0")

    # ───────────────────────────────
    # Health
    # ───────────────────────────────
    @app.get("/health")
    def health():
        return {"status": "ok"}

    # ───────────────────────────────
    # Upload
    # ───────────────────────────────
    @app.post("/upload")
    async def upload(file: UploadFile = File(...)):
        suffix = Path(file.filename).suffix

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        shutil.copyfileobj(file.file, tmp)
        tmp.close()

        dataset_id = engine.upload(tmp.name, name=file.filename)

        return {
            "dataset_id": dataset_id,
            "status": "ready"
        }

    # ───────────────────────────────
    # Query
    # ───────────────────────────────
    @app.post("/query")
    def query(req: QueryRequest):
        result = engine.query(req.dataset_id, req.query)
        return result

    # ───────────────────────────────
    # Datasets
    # ───────────────────────────────
    @app.get("/datasets")
    def datasets():
        return engine.datasets()

    # ───────────────────────────────
    # Delete
    # ───────────────────────────────
    @app.delete("/datasets/{dataset_id}")
    def delete(dataset_id: str):
        return {"deleted": engine.delete(dataset_id)}

    return app