from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
from torchvision.models import vit_b_16

torch.set_num_threads(1)

app = FastAPI(title="Smart Crop Doctor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class_names = [
    'Apple__Apple_scab','Apple_Black_rot','Apple_Cedar_apple_rust','Apple_healthy',
    'Blueberry_healthy',"Cherry(including_sour)Powdery_mildew","Cherry(including_sour)healthy",
    'Corn(maize)Cercospora_leaf_spot Gray_leaf_spot','Corn(maize)Common_rust','Corn_(maize)Northern_Leaf_Blight',
    'Corn(maize)healthy','Grape_Black_rot','Grape_Esca(Black_Measles)',
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)','Grape__healthy',
    'Orange_Haunglongbing(Citrus_greening)','Peach__Bacterial_spot','Peach_healthy',
    'Pepper,_bell_Bacterial_spot','Pepper,_bell_healthy',
    'Potato_Early_blight','Potato_Late_blight','Potato_healthy',
    'Raspberry_healthy','Soybean_healthy','Squash_Powdery_mildew',
    'Strawberry_Leaf_scorch','Strawberry_healthy',
    'Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight',
    'Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites Two-spotted_spider_mite',
    'Tomato_Target_Spot','Tomato_Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_Tomato_mosaic_virus','Tomato__healthy'
]

NUM_CLASSES = 38

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ---------------- LAZY MODEL LOAD ----------------
model = None

def load_model():
    global model
    if model is None:
        m = vit_b_16(weights=None)
        m.heads.head = torch.nn.Linear(m.heads.head.in_features, NUM_CLASSES)
        m.load_state_dict(torch.load("vit_plantdisease.pt", map_location="cpu"))
        m = m.half()
        m.eval()
        model = m
    return model

def predict(image):
    mdl = load_model()
    img = transform(image).unsqueeze(0).half()

    with torch.no_grad():
        out = mdl(img)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    return class_names[pred.item()], float(conf.item())

@app.get("/")
def home():
    return {"message": "Smart Crop Doctor API is running"}

@app.post("/predict")
async def predict_crop(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    label, confidence = predict(image)

    return {
        "disease": label,
        "confidence": round(confidence * 100, 2)
    }
