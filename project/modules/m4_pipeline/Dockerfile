# Minimal container so the pipeline runs anywhere. (Week 4)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "modules/m4_pipeline/pipeline.py"]
