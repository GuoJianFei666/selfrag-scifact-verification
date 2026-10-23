from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictInt

WEB = Path(__file__).parent / "web"


class ClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    claim: str = Field(min_length=1, max_length=5000)
    top_k: StrictInt = Field(default=5, ge=1, le=20)
    document_ids: list[StrictInt] | None = Field(default=None, min_length=1, max_length=100)


def create_app(engine):
    app = FastAPI(title="Self-RAG scientific evidence verification", version="0.1.0")

    @app.middleware("http")
    async def local_access(request: Request, call_next):
        host = request.headers.get("host", "").split(":", 1)[0]
        if host not in ("127.0.0.1", "localhost", "testserver"):
            return JSONResponse({"detail": "Local host required"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin != "http://" + request.headers.get("host", ""):
            return JSONResponse({"detail": "Cross-origin request rejected"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'"
        return response

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/health")
    def health():
        return engine.health()

    @app.post("/api/search")
    def search(payload: ClaimRequest):
        try:
            return {"claim": payload.claim, "documents": engine.search(payload.claim, payload.top_k, payload.document_ids)}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/api/verify")
    def verify(payload: ClaimRequest):
        try:
            return engine.verify(payload.claim, payload.top_k, payload.document_ids)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/api/reports/{request_id}")
    def report(request_id: str):
        try:
            value = engine.report(request_id)
            return JSONResponse(value, headers={"Content-Disposition": f'attachment; filename="evidence-{request_id}.json"'})
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404, "Report not found") from exc

    app.mount("/assets", StaticFiles(directory=WEB), name="assets")
    return app
