FROM python:3.12 AS production
WORKDIR /production
RUN pip install --no-cache-dir torch torchvision uvicorn fastapi
COPY api.py dataset.py model.py .
COPY models/FashionMNIST.pth models/FashionMNIST.pth
COPY fashion_mnist/ fashion_mnist/ 
EXPOSE 8000
CMD ["python", "api.py"]
