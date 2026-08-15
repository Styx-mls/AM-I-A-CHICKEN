FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/api.py ./src/api.py
COPY src/inference.py ./src/inference.py
COPY src/model.py ./src/model.py

COPY models/ ./models/
COPY frontend/ ./frontend/

CMD ["uvicorn", "api:api", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]