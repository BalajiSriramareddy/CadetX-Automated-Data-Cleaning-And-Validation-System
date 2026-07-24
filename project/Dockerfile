# =============================================================================
# Automated Data Quality Pipeline — container image
# =============================================================================
# Makes the pipeline reproducible: same Python, same library versions, anywhere.
#
# Build:
#   docker build -t data-quality-pipeline .
#
# Run (mount your data in, get results out):
#   docker run --rm \
#     -v "$(pwd)/data:/app/data" \
#     -v "$(pwd)/outputs:/app/outputs" \
#     data-quality-pipeline --input data/raw/broadband_customers.csv
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first, as a separate layer — Docker caches this, so
# changing application code does not force a full reinstall on every build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The pipeline exits 1 on a FAIL verdict, so this container can be used as a
# quality gate in an automated workflow.
ENTRYPOINT ["python", "modules/m4_pipeline/pipeline.py"]
CMD ["--help"]