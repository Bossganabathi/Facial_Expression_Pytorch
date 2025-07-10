from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision import datasets, transforms, models

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Original 
# Define model
# Load trained weights correctly
state_dict = torch.load(r"C:\Users\User\Documents\Machine_Learning_Project\Facial_Expression_Pytorch\model\emotion_classifier_(latest_18).pth", map_location=torch.device('cpu'))  # or 'cuda'
model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 3)

# Load the state dict into the model
model_ft.load_state_dict(state_dict)

# Set to eval mode
model_ft.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No image part'
        file = request.files['image']
        if file.filename == '':
            return 'No selected image'
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0)  # Add batch dimension

            # Predict
            with torch.no_grad():
                result = model_ft(image)
                print("Result:", result)
                predicted = torch.max(result, 1)
                class_names = ['happy', 'neutral', 'sad']
                print("Predicted:", class_names[predicted[1]])
    return render_template('index.html', prediction=class_names[predicted[1]], image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)

    