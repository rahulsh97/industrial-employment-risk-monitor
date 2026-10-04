# Industrial Employment Risk Monitor — Streamlit research preview
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# application code + frozen model + compact derived data (no raw UNIDO)
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY reports/ ./reports/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501
# Cloud Run provides $PORT; default to 8501 locally.
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
