# AutoSnap API

https://github.com/user-attachments/assets/e286b657-c86a-4ae7-8e2e-ccea673a210b

Minimal FastAPI image-inference backend. The model manager is initialized during application startup and is shared by requests.

## Run locally

The project pins Python 3.13 through `mise`, which is compatible with TensorFlow 2.20.

```bash
mise install
mise exec -- python -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/uvicorn app.main:app --reload
```

## Unified UI and API documentation

```bash
mise run demo
```

The English UI is available at `/en`, the Russian UI at `/ru`. Each UI includes a static training-notebook overview; notebook cells are displayed for reference and are not executed. OpenAPI documentation remains available at `/docs` and `/redoc`. API status is available at `/api`, and inference at `/predict`.

Check `http://localhost:8000/docs`, or send an image:

```bash
curl -X POST http://localhost:8000/predict -F "file=@image.jpg"
```

At startup, the service first uses the local bundle from `MODEL_PATH`. If that file is missing, it downloads the model files from `HF_MODEL_REPO` into `HF_CACHE_DIRECTORY` with `huggingface_hub`, then loads the model. The endpoint uses RGB conversion and `tf.image.resize_with_pad(..., 224, 224)` with float32 pixels in the original `0..255` range; it does not divide by 255 because normalization is already inside the saved EfficientNet model. The classifier's bundle metadata currently declares 196 classes. The response includes the predicted car, confidence, and a price range when the lookup data has enough listings.

## Hugging Face model storage

The trained model bundle is stored in the [Dmustache/autosnap-car-model](https://huggingface.co/Dmustache/autosnap-car-model) repository. Git contains only application code and documentation; model weights are not committed to this repository.

At startup, `ModelManager` follows this order:

1. load `MODEL_PATH` if the local bundle exists;
2. otherwise download the required model files from `HF_MODEL_REPO` at `HF_MODEL_REVISION`;
3. reuse the local Hugging Face cache on subsequent starts;
4. load `car_classifier.keras` together with its metadata and price lookup.

The public model repository does not require a token. For a private repository, authenticate once with `hf auth login` or provide `HF_TOKEN` through the environment; never store tokens in Git or `.env` files committed to the repository.

For the training notebook, install the additional research dependencies from `requirements-training.txt`, then select the Python 3.13 environment as the Jupyter kernel.

## Docker Distroless

The production image uses a multi-stage build with `gcr.io/distroless/python3-debian13:nonroot`. The model is not copied into the image; if no local model is mounted, the application downloads it from Hugging Face during startup into `/tmp/autosnap-hf-cache`.

```bash
docker build -t autosnap-api:latest .
docker run --rm -p 8000:8000 autosnap-api:latest
```

Open `http://localhost:8000/en` for the UI or `http://localhost:8000/docs` for OpenAPI. Distroless images do not include a shell; use a separate debug image when interactive container inspection is needed.

## Layout

```text
app/
  main.py                    # FastAPI app and startup/shutdown lifespan
  api/routes.py              # health and image upload endpoints
  core/config.py             # environment-backed settings
  services/model_manager.py  # model lifecycle and adapter boundary
  schemas/prediction.py      # response schemas
training/
  classification_en.ipynb    # static English training overview
  classification_ru.ipynb    # static Russian training overview
models/                      # local ignored model artifacts
```
