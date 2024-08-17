from fastapi import FastAPI
from app.api import app as fastapi_app

app = fastapi_app

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
