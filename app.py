"""
Storytelling App for Kids (aged 3-10)
======================================
An interactive Streamlit application that:
    1. Accepts an image uploaded by the user.
    2. Generates a caption using the Hugging Face image-captioning pipeline
       (Salesforce/blip-image-captioning-base).
    3. Expands the caption into a 50-100 word children's story using a
       Hugging Face text-generation pipeline (distilgpt2).
    4. Converts the story into speech using gTTS and plays it back.

Designed to be deployed on Streamlit Cloud.
"""

import io
import re

import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "distilgpt2"
MIN_WORDS = 50      # lower bound of the required story length
MAX_WORDS = 100     # upper bound of the required story length
MAX_RETRIES = 3     # retries if the first story is too short
OPENING = "Once upon a time,"   # kept at the front of every story


# ---------------------------------------------------------------------------
# Model loading (cached so models are downloaded/loaded only once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_captioner():
    """Load the pre-trained image-text-to-text (image captioning) pipeline."""
    return pipeline("image-text-to-text", model=CAPTION_MODEL)


@st.cache_resource(show_spinner=False)
def load_story_generator():
    """Load the pre-trained text-generation pipeline."""
    return pipeline("text-generation", model=STORY_MODEL)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def clean_caption(raw: str) -> str:
    """Normalise a raw BLIP caption: strip whitespace and trailing punctuation."""
    caption = raw.strip()
    caption = caption.rstrip(".!? ").strip()
    return caption


def generate_caption(captioner, image: Image.Image) -> str:
    """
    Produce a short caption describing the content of an uploaded image.

    Args:
        captioner: A Hugging Face image-text-to-text pipeline.
        image: A PIL Image object.

    Returns:
        A cleaned caption string, e.g. "a dog running in the park".
    """
    outputs = captioner(image.convert("RGB"))
    return clean_caption(outputs[0]["generated_text"])


def _split_sentences(text: str):
    """
    Split text into sentences robustly.

    GPT-2 occasionally emits sentences stuck together without a space,
    e.g. "in the park.Then a dog ran over". We insert a space after a
    sentence-ending punctuation mark when it is immediately followed by an
    uppercase letter, then split on punctuation + whitespace.
    """
    text = re.sub(r"(?<=[.!?])(?=[A-Z0-9])", " ", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _assemble_story(completion: str):
    """
    Build a story that starts with OPENING and contains between MIN_WORDS and
    MAX_WORDS words.

    Strategy:
        1. Greedily add whole sentences until MIN_WORDS is reached (the floor
           is the harder requirement when sentences are chunky).
        2. Once the floor is met, keep adding sentences only while the total
           stays at or below MAX_WORDS.
        3. As a final safety net, hard-trim at MAX_WORDS words, preferring to
           stop at the last complete sentence.

    Returns:
        (story_text, word_count)
    """
    sentences = _split_sentences(completion)
    opening_words = len(OPENING.split())

    chosen = []
    word_count = opening_words
    for sent in sentences:
        sw = len(sent.split())
        # Floor not reached yet -> keep going even if this sentence pushes
        # close to the ceiling (a final hard trim will correct it).
        if word_count >= MIN_WORDS and word_count + sw > MAX_WORDS:
            break
        chosen.append(sent)
        word_count += sw

    story = f"{OPENING} {' '.join(chosen)}".strip()

    # Hard ceiling: if a single chunky sentence pushed us over MAX_WORDS,
    # trim at the last complete sentence within the budget.
    words = story.split()
    if len(words) > MAX_WORDS:
        trimmed = words[:MAX_WORDS]
        for i in range(len(trimmed) - 1, -1, -1):
            if trimmed[i].endswith((".", "!", "?")):
                trimmed = trimmed[: i + 1]
                break
        story = " ".join(trimmed)
        word_count = len(trimmed)
    else:
        word_count = len(words)

    # Note: the continuation is intentionally left as-is after the comma in
    # OPENING ("Once upon a time, a little dog..."); lower-case there is
    # grammatically correct.
    return story, word_count


def generate_story(generator, caption: str) -> str:
    """
    Expand an image caption into a 50-100 word, child-friendly story.

    If the first generation lands outside the word range, retry with a
    slightly higher temperature; use the best result either way.

    Args:
        generator: A Hugging Face text-generation pipeline.
        caption: A cleaned caption string.

    Returns:
        A 50-100 word story as a string.
    """
    clean = clean_caption(caption)
    prompt = (
        "A short, warm children's picture-book story about "
        f"{clean}. {OPENING} "
    )

    best_story, best_wc = "", 0
    for attempt in range(MAX_RETRIES):
        result = generator(
            prompt,
            max_new_tokens=140,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7 + attempt * 0.1,
            no_repeat_ngram_size=2,
            pad_token_id=generator.tokenizer.eos_token_id,
        )
        # Remove the prompt (including its trailing OPENING) from the output;
        # OPENING is re-attached by _assemble_story so the story reads
        # cleanly from a complete phrase.
        completion = result[0]["generated_text"][len(prompt):]
        story, wc = _assemble_story(completion)

        if MIN_WORDS <= wc <= MAX_WORDS:
            return story
        # Track the closest valid-length result as a fallback.
        if best_story == "" or abs(wc - 75) < abs(best_wc - 75):
            best_story, best_wc = story, wc

    return best_story


def text_to_speech(text: str) -> io.BytesIO:
    """
    Convert a piece of text into MP3 audio using gTTS.

    The audio is kept in memory (BytesIO) so no temporary files are leaked
    to disk between Streamlit reruns.

    Args:
        text: The story to be read aloud.

    Returns:
        A BytesIO buffer positioned at the start, ready for st.audio().
    """
    tts = gTTS(text=text, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def set_page_style() -> None:
    """Inject a few simple CSS tweaks to make the UI feel child-friendly."""
    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 2rem; }
        h1 { color: #ff6b6b; text-align: center; }
        .story-box {
            background-color: #fff8e7;
            border-left: 6px solid #ffb703;
            padding: 16px;
            border-radius: 8px;
            font-size: 18px;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Storytime with Doubao",
        page_icon="📚",
        layout="centered",
    )
    set_page_style()

    st.title("📚 Storytime!")
    st.write(
        "Upload a picture and I will turn it into a magical story "
        "just for you! 🌈✨"
    )

    uploaded_file = st.file_uploader(
        "Choose a picture (a drawing, a photo, anything!)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your picture", use_container_width=True)

        if st.button("📖 Tell me a story!", type="primary"):
            with st.spinner("Looking at your picture... 🖼️"):
                try:
                    captioner = load_captioner()
                    caption = generate_caption(captioner, image)
                except Exception as exc:  # pragma: no cover - runtime guard
                    st.error(f"Sorry, I couldn't read that picture: {exc}")
                    return

            with st.spinner("Writing your story... ✍️"):
                try:
                    generator = load_story_generator()
                    story = generate_story(generator, caption)
                except Exception as exc:  # pragma: no cover - runtime guard
                    st.error(f"Oops, something went wrong: {exc}")
                    return

            st.subheader("Here is your story:")
            st.markdown(
                f'<div class="story-box">{story}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"*About {len(story.split())} words.*")

            with st.spinner("Reading it out loud... 🔊"):
                try:
                    audio_buf = text_to_speech(story)
                    st.audio(audio_buf, format="audio/mp3")
                except Exception as exc:  # pragma: no cover - runtime guard
                    st.warning(f"I couldn't read the story out loud: {exc}")

    st.markdown("---")
    st.caption("Made with ❤️ for curious kids aged 3-10.")


if __name__ == "__main__":
    main()
#（注：内容由AI生成）
