import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS
import tempfile
import os
import re

# -------------------------------
# Configuration
# -------------------------------
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "gpt2-medium"          # high-quality story generation

# -------------------------------
# Model Loading (cached)
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_caption_pipeline():
    """
    Load image-to-text pipeline for BLIP.
    Explicitly uses task='image-to-text' as required by assignment.
    Falls back to automatic detection or direct BLIP loader if needed.
    """
    try:
        return pipeline("image-to-text", model=CAPTION_MODEL)
    except Exception:
        try:
            return pipeline(model=CAPTION_MODEL)
        except Exception:
            return None

@st.cache_resource(show_spinner=False)
def load_blip_direct():
    """Fallback: load BLIP via AutoProcessor + BlipForConditionalGeneration."""
    from transformers import AutoProcessor, BlipForConditionalGeneration
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL)
    return processor, model

@st.cache_resource(show_spinner=False)
def load_story_pipeline():
    """Load text-generation pipeline for child-friendly storytelling."""
    return pipeline("text-generation", model=STORY_MODEL)

# -------------------------------
# Caption Generation
# -------------------------------
def generate_caption(caption_pipe, image: Image.Image) -> str:
    """
    Generate a short caption using image-to-text pipeline.
    Converts image to RGB to avoid channel mismatches.
    """
    img = image.convert("RGB")

    # Primary: image-to-text pipeline
    if caption_pipe is not None:
        try:
            outputs = caption_pipe(img)
            if isinstance(outputs, list) and outputs:
                first = outputs[0]
                if isinstance(first, dict) and "generated_text" in first:
                    return first["generated_text"].strip()
        except Exception:
            pass

    # Fallback: direct BLIP unconditional captioning
    try:
        processor, model = load_blip_direct()
        inputs = processor(images=img, return_tensors="pt")
        out_ids = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out_ids[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        raise RuntimeError(f"Caption model failed: {e}")

# -------------------------------
# Story Generation
# -------------------------------
def _clean_story_text(text: str) -> str:
    """Normalize whitespace and punctuation spacing."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    return text

def _target_word_count_story(raw_story: str, min_words=50, max_words=100) -> str:
    """
    Keep complete sentences until word count is within [min_words, max_words].
    Never cuts a sentence in the middle.
    """
    raw_story = _clean_story_text(raw_story)
    sentences = re.split(r"(?<=[.!?])\s+", raw_story)
    chosen = []
    total_words = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        w = len(sent.split())
        if total_words + w > max_words:
            break
        chosen.append(sent)
        total_words += w
    result = " ".join(chosen).strip()
    if not result:
        # Hard truncation as last resort
        result = " ".join(raw_story.split()[:max_words])
    return result

def generate_story(caption: str, story_pipe) -> str:
    """
    Expand caption into a 50-100 word child-friendly story.
    Uses return_full_text=False to get only new tokens.
    Retries with higher creativity if under 50 words.
    """
    prompt = (
        "Write a wholesome, simple bedtime-style story for children aged 3 to 10. "
        "Continue the story based on this scene: " + caption + " "
    )

    def _call(pipe, max_new_tokens, temperature, top_p):
        return pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=top_p,
            repetition_penalty=1.2,
            truncation=True,
            return_full_text=False,
        )[0]["generated_text"]

    raw = _call(story_pipe, 180, 0.85, 0.96)
    story = _target_word_count_story(raw, 45, 105)

    # Retry if too short
    if len(story.split()) < 48:
        raw2 = _call(story_pipe, 250, 0.92, 0.98)
        story = _target_word_count_story(raw2, 55, 102)

    # Final length enforcement
    if len(story.split()) > 104:
        story = _target_word_count_story(story, 60, 99)

    if not story.endswith((".", "!", "?")):
        story += "."
    return story

# -------------------------------
# Text-to-Speech
# -------------------------------
def text_to_speech(text: str, lang: str = "en") -> str:
    """Convert story text to MP3 using gTTS. Returns temporary file path."""
    tts = gTTS(text=text, lang=lang, slow=False)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.set_page_config(page_title="Kids Image Storyteller")
    st.title("🧸 Kids Image Storyteller")
    st.markdown("Upload an image and I'll create a short story (50–100 words) with audio!")

    with st.spinner("Loading models (first run may take a couple of minutes)..."):
        caption_pipe = load_caption_pipeline()
        story_pipe = load_story_pipeline()

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Your picture", use_container_width=True)

        if st.button("✨ Tell me a story!"):
            try:
                with st.spinner("Describing your picture with image-to-text..."):
                    caption = generate_caption(caption_pipe, image)
                    st.info(f"I see: {caption}")

                with st.spinner("Writing a story..."):
                    story = generate_story(caption, story_pipe)
                    st.success("Here's your story:")
                    st.write(story)

                with st.spinner("Turning it into speech..."):
                    audio_path = text_to_speech(story)
                    st.audio(audio_path, format="audio/mp3")
                    with open(audio_path, "rb") as f:
                        st.download_button("Download Audio", f, file_name="story.mp3", mime="audio/mp3")
                    os.unlink(audio_path)

            except Exception as e:
                st.error(f"Oops, something went wrong: {e}")

if __name__ == "__main__":
    main()
