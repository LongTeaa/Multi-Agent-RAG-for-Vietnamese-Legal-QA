"""
api/main.py
Main entry point for the FastAPI application.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.utils.logger import logger
from api.routers import qa
import time

app = FastAPI(
    title="Vietnamese Legal Multi-Agent RAG API",
    description="API for high-accuracy legal questions powered by LangGraph.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. Change in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging integration with FastAPI
@app.on_event("startup")
async def startup_event():
    logger.info("API Starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API Shutting down...")


# Root endpoint / Health check
@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "online",
        "timestamp": time.time(),
        "service": "Vietnamese Legal RAG API"
    }

# Include routers
app.include_router(qa.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
