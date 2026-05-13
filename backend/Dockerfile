FROM python:3.11-slim

WORKDIR /app

# System deps (none needed for this stack, but keep layer for future)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps — cached layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCy model — small model for free-tier RAM constraints
RUN python -m spacy download en_core_web_sm

# NLTK data (wordnet used by features.py)
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# Application code + model files
COPY . .

# Render injects PORT env var
ENV PORT=8000
EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
