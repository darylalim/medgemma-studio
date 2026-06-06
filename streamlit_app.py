import streamlit as st
from dotenv import load_dotenv
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image

load_dotenv()

MODEL_ID = "mlx-community/medgemma-1.5-4b-it-bf16"

DEFAULT_INSTRUCTION_IMAGE = "You are an expert radiologist."
DEFAULT_INSTRUCTION_TEXT = "You are a helpful medical assistant."


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
    has_image: bool, is_thinking: bool, system_instruction: str
) -> tuple[str, int]:
    if is_thinking:
        return (
            f"SYSTEM INSTRUCTION: think silently if needed. {system_instruction}",
            1300,
        )
    max_new_tokens = 300 if has_image else 500
    return system_instruction, max_new_tokens


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

    run_btn = st.button("Run", type="primary", disabled=not prompt)

    if run_btn and prompt:
        has_image = uploaded_image is not None
        full_instruction, max_new_tokens = get_generation_params(
            has_image, is_thinking, system_instruction
        )
        messages = build_messages(prompt, full_instruction, uploaded_image)
        image_for_model = [uploaded_image] if uploaded_image else None
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

        st.markdown("### Response")
        st.markdown(response)


if __name__ == "__main__":
    main()
