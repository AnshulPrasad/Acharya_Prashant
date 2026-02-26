FROM python:3.10-slim

WORKDIR /code

COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt



ENV HF_HOME=/tmp/.cache

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]