from fastapi import FastAPI
from app.api import app as fastapi_app
import uvicorn

app = fastapi_app


# uvicorn.run(app, host="0.0.0.0", port=8080)
