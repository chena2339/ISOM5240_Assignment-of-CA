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


def _assemble_story(completion: str, opening_phrase: str):
    """
    Build a story starting with opening_phrase that strictly meets 50-100 word limits.

    Strategy: greedily add COMPLETE sentences until the next one would exceed
    MAX_WORDS. Fallbacks guarantee the word-count requirement is met even if
    the model produces unpunctuated text or very short output.
    """
    sentences = _split_sentences(completion)
    opening_words = len(opening_phrase.split())

    chosen = []
    word_count = opening_words

    for sent in sentences:
        sw = len(sent.split())
        # Strictly stop adding sentences BEFORE exceeding MAX_WORDS ceiling
        if word_count + sw > MAX_WORDS:
            break
        chosen.append(sent)
        word_count += sw

    # Fallback 1: if sentence splitting failed entirely (e.g. the model
    # produced one long unpunctuated stream), at least keep the first
    # sentence-shaped chunk so the story is not only the opening phrase.
    if not chosen and sentences:
        first = sentences[0]
        chosen.append(first)
        word_count += len(first.split())

    story = f"{opening_phrase} {' '.join(chosen)}".strip()

    # Fallback 2: if complete sentences alone cannot reach MIN_WORDS,
    # pad from the raw completion text (word-level, avoid reusing the
    # words already consumed by `chosen`).
    if word_count < MIN_WORDS:
        all_words = completion.split()
        already_used = sum(len(s.split()) for s in chosen)
        need = MIN_WORDS - word_count
        padding = " ".join(all_words[already_used:already_used + need])
        if padding:
            story = f"{story} {padding}"
            word_count += len(padding.split())

    # Backup safety: ensure the story ends cleanly with punctuation
    if not story.endswith((".", "!", "?")):
        story += "."

    return story, len(story.split())


def generate_story(generator, caption: str) -> str:
    """
    Expand an image caption into a logical 50-100 word children's story with complete sentences.
    """
    clean = clean_caption(caption)

    opening_phrase = f"Once upon a time, there was {clean}."
    # NOTE: avoid "they" - the caption may describe an object (e.g. a car),
    # so use a neutral opening that introduces the adventure instead.
    prompt = f"{opening_phrase} One sunny day, an exciting adventure began."

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

    return best_story


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

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Picture", use_container_width=True)

        if st.button("📖 Generate Story", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    captioner = load_captioner()
                    caption = generate_caption(captioner, image)
                    st.info(f"**Image Caption:** {caption}")
                except Exception as exc:
                    st.error(f"Failed to process image: {exc}")
                    return

            with st.spinner("Writing story..."):
                try:
                    generator = load_story_generator()
                    story = generate_story(generator, caption)
                except Exception as exc:
                    st.error(f"Failed to generate story: {exc}")
                    return

            st.subheader("Generated Story:")
            st.success(story)
            st.caption(f"*Word count: {len(story.split())} words*")

            with st.spinner("Generating audio..."):
                try:
                    audio_buf = text_to_speech(story)
                    st.audio(audio_buf, format="audio/mp3")
                except Exception as exc:
                    st.warning(f"Failed to generate audio: {exc}")

    st.markdown("---")
    st.caption("Designed for children aged 3-10.")


if __name__ == "__main__":
    main()
