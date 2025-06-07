FROM python:3.10-slim

WORKDIR /code

COPY . .

RUN pip install --no-cache-dir -r data/requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]