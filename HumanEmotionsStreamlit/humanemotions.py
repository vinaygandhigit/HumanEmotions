import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

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
    
@st.cache_resource
def model_load():
    model = FER_CNN()
    model.load_state_dict(torch.load('HumanEmotions.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = transform(image).unsqueeze(0)
    return image

def predict_emotions(model, image):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_idx = torch.argmax(outputs, dim=1).item()
        predicted_emotions = emotions[predicted_idx]

    return predicted_emotions, probabilities

st.title("Facial Expression Recognition with Custom CNN")
st.title("Upload an image to predict the emotion(angry, disgust, fear, happy, neutral, sad, surprise)")

uploaded_file = st.file_uploader("Choose an Image...", type=["jpg","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    try:
        model = model_load()
        proccessed_image = preprocess_image(image)
        predicted_emotions, probabilities = predict_emotions(model, proccessed_image)
        st.write(f"**Predicted Emotion : { predicted_emotions}" )
        st.write(f"Confidence Score")

        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        for emotion, prob in zip(emotions, probabilities):
            st.write(f"{emotion}: {prob:.4f}")
        
        # Bar plot for probabilities
        st.bar_chart(dict(zip(emotions, probabilities)))

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.write("Please upload an image to get a prediction.")
    
 
    