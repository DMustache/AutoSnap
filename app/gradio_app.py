"""Localized Gradio UIs for trying the AutoSnap model."""

import json
from pathlib import Path

import gradio as gr
from PIL import Image

from app.services.model_manager import ModelManager

print("Gradio version:", gr.__version__)


def render_notebook_overview(language: str) -> str:
    """Render notebook markdown and code as static documentation; never execute it."""
    notebook_name = "classification_en.ipynb" if language == "en" else "classification_ru.ipynb"
    notebook_path = Path(__file__).resolve().parent.parent / notebook_name
    if not notebook_path.exists():
        return f"Notebook `{notebook_name}` is not available."

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sections = []
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", [])).strip()
        if not source:
            continue
        if cell.get("cell_type") == "markdown":
            sections.append(source)
        elif cell.get("cell_type") == "code":
            sections.append(f"```python\n{source}\n```")

    heading = "Training overview" if language == "en" else "Обзор обучения"
    note = (
        "This is a static overview of the training notebook. Cells are shown for reference and are not executed."
        if language == "en"
        else "Это статический обзор обучающего ноутбука. Ячейки показаны только для ознакомления и не выполняются."
    )
    return f"## {heading}\n\n{note}\n\n" + "\n\n---\n\n".join(sections)


def _format_price(price: dict | None) -> str:
    if not price:
        return "Unavailable"
    return f"${price['lower']:,.0f}–${price['upper']:,.0f} (median ${price['median']:,.0f})"


def _format_top5(top5: list[dict]) -> str:
    return "\n".join(f"{item['label']}: {item['confidence']:.1%}" for item in top5)


def predict_for_interface(image: Image.Image | None, manager: ModelManager, language: str = "en"):
    texts = {
        "en": {"upload": "Upload an image", "unavailable": "Model unavailable", "unknown": "Unknown / unsupported", "recognized": "Recognized", "price_unavailable": "Unavailable"},
        "ru": {"upload": "Загрузите изображение", "unavailable": "Модель недоступна", "unknown": "Неизвестно / класс не поддерживается", "recognized": "Распознано", "price_unavailable": "Недоступно"},
    }[language]
    if image is None:
        return texts["upload"], "", "", "", "", "", ""
    if not manager.is_loaded:
        return f"{texts['unavailable']}: {manager.load_error}", "", "", "", "", "", ""

    result = manager.predict(image)
    if not result["known"]:
        return (
            f"{texts['unknown']} ({result['confidence']:.1%})", "", "", "",
            texts["price_unavailable"], f"{result['confidence']:.1%}",
            _format_top5(result["top5"]),
        )
    return (
        texts["recognized"], result["make"], result["model"], str(result["year"]),
        _format_price(result["price"]), f"{result['confidence']:.1%}",
        _format_top5(result["top5"]),
    )


def create_demo(manager: ModelManager, language: str = "en") -> gr.Interface:
    """Build an English or Russian UI around the shared model manager."""
    if language == "ru":
        labels = {
            "input": "Загрузите фото автомобиля", "status": "Статус", "make": "Марка",
            "model": "Модель", "year": "Год", "price": "Ожидаемая стоимость, USD",
            "confidence": "Уверенность", "top5": "Top-5 предсказаний",
            "title": "Распознавание автомобиля и оценка стоимости",
            "description": "Загрузите фотографию одного автомобиля. Система распознаёт классы Stanford Cars 196. Стоимость — приблизительная медиана объявлений, а не рыночная оценка.",
        }
    else:
        labels = {
            "input": "Upload a car photo", "status": "Status", "make": "Make",
            "model": "Model", "year": "Year", "price": "Estimated price, USD",
            "confidence": "Confidence", "top5": "Top-5 predictions",
            "title": "Car Recognition and Price Estimation",
            "description": "Upload a photo containing one clearly visible car. The system recognizes classes represented in Stanford Cars 196. Price is an approximate median listing value, not a market appraisal.",
        }
    return gr.Interface(
        fn=lambda image: predict_for_interface(image, manager, language),
        inputs=gr.Image(type="pil", label=labels["input"]),
        outputs=[
            gr.Textbox(label=labels["status"]), gr.Textbox(label=labels["make"]),
            gr.Textbox(label=labels["model"]), gr.Textbox(label=labels["year"]),
            gr.Textbox(label=labels["price"]), gr.Textbox(label=labels["confidence"]),
            gr.Textbox(label=labels["top5"], lines=6),
        ],
        title=labels["title"],
        description=labels["description"],
        article=render_notebook_overview(language),
        flagging_mode="never",
    )


if __name__ == "__main__":
    from app.core.config import get_settings

    standalone_manager = ModelManager(get_settings())
    standalone_manager.load()
    create_demo(standalone_manager, language="en").launch(share=False, debug=False)
