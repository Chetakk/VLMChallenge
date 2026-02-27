"""
API Main Module

FastAPI application for VLM inference API.
"""

from fastapi import FastAPI

app = FastAPI(title="VLM Challenge API")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict")
async def predict(image_data: str, prompt: str):
    """Run inference on image with given prompt."""
    pass


@app.get("/models")
async def list_models():
    """List available models."""
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
