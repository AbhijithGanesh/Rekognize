from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
from src import run_model
import uuid
from facenet_pytorch import MTCNN, extract_face
import torch

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# mtcnn, resnet = run_model.load_models()
IMAGEDIR = "./test_images/"
OUTDIR = "classified_images"

app = FastAPI()


@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()  # <-- Important!
    # example of how you can save the file
    with open(f"{IMAGEDIR}/{file.filename}", "wb") as f:
        f.write(contents)

    return {"filename": file.filename}


@app.get("/images/{image_id}")
async def read_file(image_id: int):
    files = os.listdir(IMAGEDIR)

    if image_id < len(files):
        return FileResponse(f'{IMAGEDIR}/{files[image_id]}')
    else:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")


@app.get("/get-attendance/{image_id}")
async def detect_faces(image_id: int = 0):
    files = os.listdir(IMAGEDIR)

    if image_id < len(files):
        run_model.run_all(f'{IMAGEDIR}/{files[image_id]}', './standard_set')
        return FileResponse(f'output.png')
    else:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
