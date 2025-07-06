import torch
from PIL import Image
from torchvision import transforms
import timm
import torch.nn as nn

from torchvision import datasets, transforms, models

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io

# Define model
# Load trained weights correctly
state_dict = torch.load(r"C:\Users\User\Documents\Machine_Learning_Project\House_Price_Prediction_-CI-CD-\model\emotion_classifier_(latest_18).pth", map_location=torch.device('cpu'))  # or 'cuda'
model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 3)

# Load the state dict into the model
model_ft.load_state_dict(state_dict)

# Set to eval mode
model_ft.eval()

# Define transform (with normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

app = FastAPI()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(content={"error": "Invalid image format"}, status_code=400)

# Load and preprocess image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]
        
    with torch.no_grad():
        result = model_ft(input_tensor)
        print("Result:", result)
        predicted = torch.max(result, 1)
        class_names = ['happy', 'neutral', 'sad']
        print("Predicted:", class_names[predicted[1]])
        return {"filename": file.filename, "result": class_names[predicted[1].item()]}