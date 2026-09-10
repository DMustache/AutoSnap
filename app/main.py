from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.gradio_app import create_demo
from app.services.model_manager import ModelManager


model_manager = ModelManager(get_settings())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the process starts and release it on shutdown."""
    model_manager.load()
    app.state.model_manager = model_manager
    yield
    model_manager.unload()


app = FastAPI(
    title="AutoSnap API",
    version="0.1.0",
    description="Image inference backend for the AutoSnap trained model.",
    lifespan=lifespan,
)
app.include_router(router)
app = gr.mount_gradio_app(app, create_demo(model_manager, language="en"), path="/en")
app = gr.mount_gradio_app(app, create_demo(model_manager, language="ru"), path="/ru")
