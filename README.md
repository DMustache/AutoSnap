# AutoSnap API

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

The included bundle is loaded at startup. The endpoint uses RGB conversion and `tf.image.resize_with_pad(..., 224, 224)` with float32 pixels in the original `0..255` range; it does not divide by 255 because normalization is already inside the saved EfficientNet model. The classifier's bundle metadata currently declares 196 classes. The response includes the predicted car, confidence, and a price range when the lookup data has enough listings.

For the training notebook, install the additional research dependencies from `requirements-training.txt`, then select the Python 3.13 environment as the Jupyter kernel.

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
