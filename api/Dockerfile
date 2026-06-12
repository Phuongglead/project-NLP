# GPU image: PyTorch + CUDA pre-installed (avoids pip pulling ~2GB nvidia-* wheels)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV NVIDIA_VISIBLE_DEVICES=all

COPY requirements-docker.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY main.py .
COPY config ./config
COPY src ./src
COPY api ./api
COPY best_model ./best_model
COPY models ./models
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
