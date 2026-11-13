FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOME=/home/app \
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    FASTEMBED_CACHE_PATH=/cache/fastembed \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home app \
    && mkdir /cache \
    && chown app:app /cache

WORKDIR /app
COPY requirements.txt ./requirements.txt
COPY streamlit-app/requirements.txt ./streamlit-app/requirements.txt
COPY deploy/requirements.txt ./deploy/requirements.txt
RUN python -m pip install --no-cache-dir -r deploy/requirements.txt

COPY src/ ./src/
COPY streamlit-app/app.py ./streamlit-app/app.py
COPY streamlit-app/style.css ./streamlit-app/style.css
COPY streamlit-app/fonts/ ./streamlit-app/fonts/
COPY streamlit-app/.streamlit/ ./streamlit-app/.streamlit/
USER app
WORKDIR /app/streamlit-app
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=4 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).read()"]
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
