from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI()
ROOT = Path("outputs").resolve()
ALLOWED = {"package","video","poster","mockup","final"}

@app.get("/ai/products/{product_id}/images/{img_type}")
def get_img(product_id: int, img_type: str):
    if img_type not in ALLOWED:
        raise HTTPException(400, "invalid type")

    path = ROOT / "products" / str(product_id) / f"{img_type}.png"
    if not path.exists():
        raise HTTPException(404, "not found")

    return FileResponse(path, media_type="image/png")

@app.get("/ai/hello")
def hello():
    return {"message": "hello"}