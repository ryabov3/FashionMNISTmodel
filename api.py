import argparse
import random

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from dataset import test_dataset
from model import FashionMNISTModel

app = FastAPI()

checkpoint = torch.load("models/FashionMNIST.pth")
model = FashionMNISTModel()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

@app.get("/")
async def start():
    response = {
        "Description": "FashionMNIST model API",
        "Random predict": ["/predict"]
    }
    return JSONResponse(content=response)

@app.get("/predict")
async def get_random_sample():
    random_idx = random.randint(0, len(test_dataset) - 1)
    image, label = test_dataset[random_idx]
    
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        predicted = torch.argmax(output, dim=1).item()

    response = {
        "true_label": class_names[label],
        "predicted_label": class_names[predicted],
        "index": random_idx
    }
    
    return JSONResponse(content=response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastAPI server with specified port")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)