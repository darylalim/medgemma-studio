import json
import re

import streamlit as st
from dotenv import load_dotenv
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image, ImageDraw

load_dotenv()

MODEL_ID = "mlx-community/medgemma-1.5-4b-it-bf16"

DEFAULT_INSTRUCTION_IMAGE = "You are an expert radiologist."
DEFAULT_INSTRUCTION_TEXT = "You are a helpful medical assistant."

LOCALIZATION_INSTRUCTION = (
    "You are an expert radiologist localizing anatomy on a medical image. "
    "Return only a JSON list inside a ```json code block. Each item must be "
    '{"box_2d": [y0, x0, y1, x1], "label": "<structure>"}, where (y0, x0) is the '
    "top-left corner and (y1, x1) the bottom-right corner, each normalized to the "
    'range [0, 1000]. "left" and "right" refer to the patient\'s anatomical sides.'
)


@st.cache_resource
def load_model():
    model, processor = load(MODEL_ID)
    config = load_config(MODEL_ID)
    return model, processor, config


def parse_response(response: str, is_thinking: bool) -> tuple[str | None, str]:
    if is_thinking and "<unused95>" in response:
        thought, answer = response.split("<unused95>", 1)
        thought = thought.removeprefix("<unused94>thought\n")
        return thought, answer
    return None, response


def build_messages(
    prompt: str, system_instruction: str, image: Image.Image | None
) -> list:
    user_content: list[dict] = [{"type": "text", "text": prompt}]
    if image is not None:
        user_content.append({"type": "image"})
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content},
    ]


def get_generation_params(
    has_image: bool,
    is_thinking: bool,
    system_instruction: str,
    is_localizing: bool = False,
) -> tuple[str, int]:
    if is_localizing:
        return LOCALIZATION_INSTRUCTION, 1300 if is_thinking else 1000
    if is_thinking:
        return (
            f"SYSTEM INSTRUCTION: think silently if needed. {system_instruction}",
            1300,
        )
    max_new_tokens = 300 if has_image else 500
    return system_instruction, max_new_tokens


def pad_to_square(image: Image.Image) -> Image.Image:
    """Pad an image to an RGB square (top-left aligned) so localization
    coordinates, normalized over a square frame, map back without an offset."""
    image = image.convert("RGB")
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    padded = Image.new("RGB", (size, size))
    padded.paste(image, (0, 0))
    return padded


def parse_boxes(response: str) -> list[dict]:
    """Extract bounding boxes from a model response.

    Accepts an optional ```json fence (or a bare JSON list) and returns a list of
    {"box_2d": [y0, x0, y1, x1], "label": str}. Malformed items are dropped; an
    unparseable response yields [].
    """
    fence = re.search(r"```(?:json)?\s*(.*?)```", response, re.DOTALL)
    if fence:
        payload = fence.group(1)
    else:
        start, end = response.find("["), response.rfind("]")
        payload = response[start : end + 1] if start != -1 and end > start else response
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    boxes: list[dict] = []
    for item in data:
        box = item.get("box_2d") if isinstance(item, dict) else None
        # Require 4 numeric coords (bools excluded): scale_box divides on these,
        # and it runs outside the inference try/except, so bad values would crash.
        if (
            isinstance(box, list)
            and len(box) == 4
            and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in box
            )
        ):
            boxes.append({"box_2d": box, "label": item.get("label", "")})
    return boxes


def scale_box(box_2d: list, size: int) -> tuple[int, int, int, int]:
    """Convert a [y0, x0, y1, x1] box normalized to [0, 1000] into pixel
    (x0, y0, x1, y1) corners for a square image of side ``size``. Corners are
    ordered (x0 <= x1, y0 <= y1) so a model box with swapped corners still draws
    instead of raising in ImageDraw.rectangle."""
    y0, x0, y1, x1 = box_2d
    px0, px1 = sorted((round(x0 / 1000 * size), round(x1 / 1000 * size)))
    py0, py1 = sorted((round(y0 / 1000 * size), round(y1 / 1000 * size)))
    return px0, py0, px1, py1


def draw_boxes(image: Image.Image, boxes: list[dict]) -> Image.Image:
    """Draw labeled boxes onto a copy of ``image`` (assumed square)."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for box in boxes:
        x0, y0, x1, y1 = scale_box(box["box_2d"], annotated.width)
        draw.rectangle((x0, y0, x1, y1), outline="red", width=3)
        if box.get("label"):
            draw.text((x0 + 4, y0 + 4), box["label"], fill="red")
    return annotated


def show_response(response: str) -> None:
    st.markdown("### Response")
    st.markdown(response)


st.set_page_config(page_title="MedGemma Pipeline")


def main():
    st.title("MedGemma Pipeline")

    with st.spinner("Loading model..."):
        model, processor, config = load_model()

    prompt = st.text_input(
        "Enter your question", placeholder="e.g. Describe this X-ray"
    ).strip()

    uploaded_image = None
    uploaded_file = st.file_uploader(
        "Upload a medical image (optional)", type=["png", "jpg", "jpeg", "webp"]
    )
    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded image", width="stretch")
        except Exception:
            st.error("Failed to load image. Please upload a valid image file.")

    default_instruction = (
        DEFAULT_INSTRUCTION_IMAGE
        if uploaded_image is not None
        else DEFAULT_INSTRUCTION_TEXT
    )
    # Auto-switch the default only while the user has not edited the field, so a
    # custom instruction survives uploading/removing an image. The keyed widget's
    # state is otherwise decoupled from the changing default.
    if not st.session_state.get("system_instruction_touched"):
        st.session_state["system_instruction"] = default_instruction

    def _mark_touched():
        st.session_state["system_instruction_touched"] = True

    system_instruction = st.text_area(
        "System instruction",
        key="system_instruction",
        height=100,
        on_change=_mark_touched,
    )
    is_thinking = st.toggle("Thinking", value=False)
    is_localizing = st.toggle(
        "Locate anatomy (bounding boxes)",
        value=False,
        disabled=uploaded_image is None,
        help="Outline anatomy on the image with bounding boxes. Requires an image.",
    )
    if is_localizing and uploaded_image is not None:
        st.caption(
            "ℹ️ Localization uses a built-in prompt; the System instruction above "
            "is ignored in this mode."
        )

    run_btn = st.button("Run", type="primary", disabled=not prompt)

    if run_btn and prompt:
        has_image = uploaded_image is not None
        localize = is_localizing and has_image
        localize_size: tuple[int, int] | None = None
        full_instruction, max_new_tokens = get_generation_params(
            has_image, is_thinking, system_instruction, is_localizing=localize
        )

        if localize and uploaded_image is not None:
            # Pad to a square so the model's [0, 1000] coordinates map back without
            # an offset, then crop the annotated result to the original size.
            localize_size = uploaded_image.size
            model_image = pad_to_square(uploaded_image)
        else:
            model_image = uploaded_image

        messages = build_messages(prompt, full_instruction, model_image)
        image_for_model = [model_image] if model_image is not None else None
        num_images = 1 if image_for_model else 0
        with st.spinner("Generating response..."):
            try:
                formatted_prompt = apply_chat_template(
                    processor, config, messages, num_images=num_images
                )
                output = generate(
                    model,
                    processor,
                    formatted_prompt,  # ty: ignore[invalid-argument-type]
                    image_for_model,  # ty: ignore[invalid-argument-type]
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                    verbose=False,
                )
                response = output.text
            except Exception as e:
                st.error(f"Inference failed: {e}")
                return

        thought, response = parse_response(response, is_thinking)
        if thought is not None:
            with st.expander("Thinking trace"):
                st.markdown(thought)

        if localize:
            boxes = parse_boxes(response)
            # localize_size/model_image are always set when localize is True; the
            # explicit checks also narrow them from Optional for the type checker.
            if boxes and localize_size is not None and model_image is not None:
                width, height = localize_size
                annotated = draw_boxes(model_image, boxes).crop((0, 0, width, height))
                st.image(annotated, caption="Localized anatomy", width="stretch")
                st.markdown("### Detected structures")
                st.markdown(
                    "\n".join(
                        f"- **{box['label'] or 'unlabeled'}**: {box['box_2d']}"
                        for box in boxes
                    )
                )
            else:
                st.warning("No bounding boxes were returned.")
                show_response(response)
        else:
            show_response(response)


if __name__ == "__main__":
    main()
