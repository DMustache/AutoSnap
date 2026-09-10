FROM python:3.13-slim-trixie AS builder

WORKDIR /build
COPY requirements.txt .

RUN python -m pip install --no-cache-dir --target=/opt/python-packages -r requirements.txt


FROM gcr.io/distroless/python3-debian13:nonroot

WORKDIR /app

COPY --from=builder /opt/python-packages /opt/python-packages
COPY app ./app
COPY training ./training

ENV PYTHONPATH=/opt/python-packages
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/models/car_models_fastapi_bundle_v1.zip
ENV HF_MODEL_REPO=Dmustache/autosnap-car-model
ENV HF_MODEL_REVISION=main
ENV HF_CACHE_DIRECTORY=/tmp/autosnap-hf-cache

EXPOSE 8000

CMD ["-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
