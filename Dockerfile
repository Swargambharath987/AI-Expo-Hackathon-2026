FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image. Otherwise every cold start
# downloads it from Hugging Face, slowing the first search — and on Cloud Run
# the download lands in the in-memory filesystem, eating into the 1Gi limit.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# vectorstore/ is the pre-built search index committed to the repo
# (see ingest.py); the app only reads it.
COPY . .
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Cloud Run (and Render) inject $PORT at runtime; 8080 matches Cloud Run's
# default and is only used when nothing sets it (e.g. a plain `docker run`).
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
