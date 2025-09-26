FROM python:3.12 AS production
WORKDIR /production
RUN pip install --no-cache-dir torch torchvision uvicorn fastapi
COPY /api /api
COPY dataset.py model.py /api/
COPY models/FashionMNIST.pth models/FashionMNIST.pth
COPY fashion_mnist/ fashion_mnist/ 
EXPOSE 8000
ENTRYPOINT ["python", "/api/api.py"]
