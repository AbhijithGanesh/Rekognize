from fastapi import FastAPI, UploadFile
from PIL import Image
from io import BytesIO
from facenet_pytorch import MTCNN, extract_face

app = FastAPI()

mtcnn = MTCNN(keep_all=True, landmarks=True)

@app.post("/segment-images")
async def segment_images(image: UploadFile):
    contents = await image.read()
    image_data = BytesIO(contents)
    image_pil = Image.open(image_data)
    boxes, probs = mtcnn.detect(image_pil)
    faces = extract_face(image_pil, boxes)
    return faces
