FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download NLTK assets used by the app (tokenizer, stopwords, VADER).
RUN python -m nltk.downloader punkt punkt_tab stopwords vader_lexicon

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]

