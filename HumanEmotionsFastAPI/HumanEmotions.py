from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import io

app = FastAPI(title="Facial Expression Recognition API", description="Predict emotions from images using a custom CNN model")
trained_model = None

class FER_CNN(nn.Module):
    def __init__(self):
        super(FER_CNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2,2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2,2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*3*3,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,256),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,7)
        )

    def forward(self, x):
        x = self.network(x)
        x = self.fc_layers(x)

        return x

try:
    model = FER_CNN()
    model.load_state_dict(torch.load('HumanEmotions.pth', map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict_emotion(model, image):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_idx = torch.argmax(outputs, dim=1).item()
        predicted_emotion = emotions[predicted_idx]
    
    return predicted_emotion, probabilities.tolist()

@app.post("/predictEmotions")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and validate image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image (jpg, png, jpeg)")
        
        # Load image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess and predict
        processed_image = preprocess_image(image)
        predicted_emotion, probabilities = predict_emotion(model, processed_image)
        
        # Prepare response
        response = {
            "predicted_emotion": predicted_emotion,
            "confidence_scores": {emotion: prob for emotion, prob in zip(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], probabilities)}
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Facial Expression Recognition API. Use POST /predict to upload an image."}