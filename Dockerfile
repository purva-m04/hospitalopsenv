FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV USE_HEURISTIC=0
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app_server:app", "--host", "0.0.0.0", "--port", "7860"]
