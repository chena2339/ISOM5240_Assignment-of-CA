"""
Storytelling App for Kids (aged 3-10)
======================================
An interactive Streamlit application that:
    1. Accepts an image uploaded by the user.
    2. Generates a caption using the Hugging Face image-captioning pipeline
       (Salesforce/blip-image-captioning-base).
    3. Expands the caption into a 50-100 word children's story using a
       lightweight kids-story model (roneneldan/TinyStories-33M).
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
STORY_MODEL = "roneneldan/TinyStories-33M"
MIN_WORDS = 50      # Lower bound of required story word count
MAX_WORDS = 100     # Upper bound of required story word count
MAX_RETRIES = 3     # Retry count if generated text length is out of range


# ---------------------------------------------------------------------------
# Model Loading (Cached to prevent memory leak and slow load times)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_captioner():
    """Load the pre-trained image-to-text pipeline explicitly per assignment requirements."""
    return pipeline("image-to-text", model=CAPTION_MODEL)


@st.cache_resource(show_spinner=False)
def load_story_generator():
    """Load the pre-trained text-generation pipeline."""
    return pipeline("text-generation", model=STORY_MODEL)


# ---------------------------------------------------------------------------
# Core Logic Functions
# ---------------------------------------------------------------------------
def clean_caption(raw: str) -> str:
    """Normalise raw BLIP caption: strip spaces and trailing punctuation."""
    caption = raw.strip()
    caption = caption.rstrip(".!? ").strip()
    return caption


def generate_caption(captioner, image: Image.Image) -> str:
    """
    Generate a short caption describing the input image.

    Args:
        captioner: Hugging Face image-to-text pipeline.
        image: PIL Image object.

    Returns:
        Cleaned caption string (e.g. "a family walking in the park").
    """
    outputs = captioner(image.convert("RGB"))
    return clean_caption(outputs[0]["generated_text"])


def _split_sentences(text: str):
    """
    Split completion text into clean, complete sentences robustly.
    Insert a space after ANY sentence ender followed directly by another
    character (small models often omit the space, e.g. "sat.They saw...").
    """
    text = re.sub(r"(?<=[.!?])(?=\S)", " ", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    # Only keep complete sentences that end with punctuation
    valid_sentences = []
    for p in parts:
        p_str = p.strip()
        if p_str and p_str[-1] in [".", "!", "?"]:
            valid_sentences.append(p_str)
    return valid_sentences


# Complete sentences used only when the model output is too short.
KIND_ENDING = (
    "Friends smiled, took turns, and shared a snack. "
    "When the sky turned pink, everyone went home happy. The end."
)


def _word_count(text: str) -> int:
    return len(text.split())


def _fit_word_range(story: str) -> str:
    """Force the story into 50–100 words using complete sentences only."""
    if not story.endswith((".", "!", "?")):
        story += "."

    # Too long: drop trailing sentences until we are at or under MAX_WORDS.
    sentences = _split_sentences(story)
    if not sentences:
        sentences = [story]
    while len(sentences) > 1 and _word_count(" ".join(sentences)) > MAX_WORDS:
        sentences.pop()
    story = " ".join(sentences).strip()

    # Too short: append kind, complete sentences until we reach MIN_WORDS.
    for extra in _split_sentences(KIND_ENDING):
        if _word_count(story) >= MIN_WORDS:
            break
        candidate = f"{story} {extra}".strip()
        if _word_count(candidate) <= MAX_WORDS:
            story = candidate

    # Last resort: still short after the ending, or one leftover long sentence.
    wc = _word_count(story)
    if wc < MIN_WORDS:
        story = f"{story} {KIND_ENDING}".strip()
        story = " ".join(story.split()[:MAX_WORDS])
        if not story.endswith((".", "!", "?")):
            story += "."
    elif wc > MAX_WORDS:
        story = " ".join(story.split()[:MAX_WORDS])
        if not story.endswith((".", "!", "?")):
            story += "."

    return story


def _assemble_story(completion: str, opening_phrase: str):
    """
    Build a story starting with opening_phrase that strictly meets 50-100 word limits.

    Strategy: greedily add COMPLETE sentences until the next one would exceed
    MAX_WORDS, then pad or trim with complete sentences (never raw word slices).
    """
    sentences = _split_sentences(completion)
    opening_words = _word_count(opening_phrase)

    chosen = []
    word_count = opening_words

    for sent in sentences:
        sw = _word_count(sent)
        if word_count + sw > MAX_WORDS:
            break
        chosen.append(sent)
        word_count += sw

    # If sentence splitting failed, keep the first punctuated chunk if any.
    if not chosen and sentences:
        chosen.append(sentences[0])

    story = f"{opening_phrase} {' '.join(chosen)}".strip()
    story = _fit_word_range(story)
    return story, _word_count(story)


def generate_story(generator, caption: str) -> str:
    """
    Expand an image caption into a logical 50-100 word children's story with complete sentences.
    """
    clean = clean_caption(caption)

    # Prompt and final opening must be the same text, otherwise the model
    # continues a sentence that is later deleted and the story breaks.
    # Avoid "they": the caption may describe an object, not people.
    opening_phrase = (
        f"Once upon a time, there was {clean}. "
        "One sunny day, an exciting adventure began."
    )
    prompt = opening_phrase

    best_story, best_wc = "", 0
    for attempt in range(MAX_RETRIES):
        result = generator(
            prompt,
            max_new_tokens=140,   # Increased token limit to allow sentence completion
            do_sample=True,
            top_k=25,
            top_p=0.85,
            temperature=0.4 + attempt * 0.1,
            no_repeat_ngram_size=2,
            pad_token_id=generator.tokenizer.eos_token_id,
        )

        # Extract generated completion text
        completion = result[0]["generated_text"][len(prompt):]

        # Assemble story using only full, complete sentences
        story, wc = _assemble_story(completion, opening_phrase=opening_phrase)

        if MIN_WORDS <= wc <= MAX_WORDS:
            return story
        if best_story == "" or abs(wc - 75) < abs(best_wc - 75):
            best_story, best_wc = story, wc

    # Guarantee the required range even if every sample was awkward.
    return _fit_word_range(best_story or opening_phrase)


def text_to_speech(text: str) -> io.BytesIO:
    """Convert text into MP3 audio using gTTS in memory."""
    tts = gTTS(text=text, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Streamlit Interface
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Kids Storyteller",
        page_icon="📚",
        layout="centered",
    )

    st.title("📚 Kids Image Storyteller")
    st.write("Upload an image to generate a 50–100 word magical story with audio!")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
    )

    # Keep caption / story / audio across reruns (audio player triggers a rerun).
    if "last_file" not in st.session_state:
        st.session_state.last_file = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Picture", use_container_width=True)

        # New picture → drop the previous story so results stay in sync.
        if uploaded_file.name != st.session_state.last_file:
            st.session_state.last_file = uploaded_file.name
            for key in ("caption", "story", "audio_bytes"):
                st.session_state.pop(key, None)

        if st.button("📖 Generate Story", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    captioner = load_captioner()
                    st.session_state.caption = generate_caption(captioner, image)
                except Exception as exc:
                    st.error(f"Failed to process image: {exc}")
                    return

            with st.spinner("Writing story..."):
                try:
                    generator = load_story_generator()
                    st.session_state.story = generate_story(
                        generator, st.session_state.caption
                    )
                except Exception as exc:
                    st.error(f"Failed to generate story: {exc}")
                    return

            with st.spinner("Generating audio..."):
                try:
                    audio_buf = text_to_speech(st.session_state.story)
                    st.session_state.audio_bytes = audio_buf.getvalue()
                except Exception as exc:
                    st.session_state.pop("audio_bytes", None)
                    st.warning(f"Failed to generate audio: {exc}")

        if "caption" in st.session_state:
            st.info(f"**Image Caption:** {st.session_state.caption}")
        if "story" in st.session_state:
            st.subheader("Generated Story:")
            st.success(st.session_state.story)
            st.caption(
                f"*Word count: {_word_count(st.session_state.story)} words*"
            )
        if "audio_bytes" in st.session_state:
            st.audio(st.session_state.audio_bytes, format="audio/mp3")

    st.markdown("---")
    st.caption("Designed for children aged 3-10.")


if __name__ == "__main__":
    main()
