import logging

from fastapi import FastAPI
from app.api import routes

logger = logging.getLogger("app")

# Initialize FastAPI app with the lifespan manager
app = FastAPI(docs_url="/docs")
app.include_router(routes.router, prefix="/api")

# Run FastAPI with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
