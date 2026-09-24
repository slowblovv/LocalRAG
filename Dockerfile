FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY evaluation /app/evaluation
COPY demo_corpus /app/demo_corpus
RUN pip install --no-cache-dir -e '.[ml]'
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn","localrag.api:app","--host","0.0.0.0","--port","8000"]
