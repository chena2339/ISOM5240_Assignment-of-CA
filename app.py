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
STORY_MODEL = "gpt2-medium"  # higher quality than gpt2, no downgrade

# -------------------------------
# Model loading (cached)
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_caption_pipeline():
    """
    Load image-to-text captioning pipeline using BLIP base.
    Pinned transformers<5 keeps 'image-to-text' task name stable.
    If explicit task fails, fall back to auto task detection,
    then to a direct BLIP processor/model loader.
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
    """Fallback BLIP loader using AutoProcessor + BlipForConditionalGeneration."""
    from transformers import AutoProcessor, BlipForConditionalGeneration
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL)
    return processor, model

@st.cache_resource(show_spinner=False)
def load_story_pipeline():
    """Load text-generation pipeline for child-friendly storytelling."""
    return pipeline("text-generation", model=STORY_MODEL)

# -------------------------------
# Caption functions
# -------------------------------
def generate_caption(caption_pipe, image: Image.Image) -> str:
    """
    Generate one short caption from an image using image-to-text pipeline.
    image.convert('RGB') avoids RGBA/grayscale shape errors.
    """
    img = image.convert("RGB")

    # Primary: image-to-text pipeline
    if caption_pipe is not None:
        try:
            outputs = caption_pipe(img)
            # Standard BLIP output: [{'generated_text': '...'}]
            if isinstance(outputs, list) and outputs:
                first = outputs[0]
                if isinstance(first, dict) and "generated_text" in first:
                    return first["generated_text"].strip()
                if isinstance(first, list) and first and "generated_text" in first[0]:
                    return first[0]["generated_text"].strip()
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
# Story functions
# -------------------------------
def _clean_story_text(text: str) -> str:
    """Remove extra whitespace and fix basic punctuation spacing."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    return text

def _target_word_count_story(raw_story: str, min_words=50, max_words=100) -> str:
    """
    Keep complete sentences until word count is within min_words..max_words.
    Avoids cutting children's story in the middle of a sentence.
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
        # Fallback: hard truncation at max_words
        result = " ".join(raw_story.split()[:max_words])
    return result

def generate_story(caption: str, story_pipe) -> str:
    """
    Expand caption into a 50-100 word child-friendly story.
    Uses return_full_text=False so only newly generated text is returned.
    Retries with different sampling if story is too short.
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

    raw = _call(story_pipe, 200, 0.8, 0.95)
    story = _target_word_count_story(raw, 50, 100)

    # If still below 50 words, retry with more creative sampling
    if len(story.split()) < 50:
        raw2 = _call(story_pipe, 260, 0.9, 0.97)
        story = _target_word_count_story(raw2, 50, 100)

    # Safety cap: never exceed 100 words
    if len(story.split()) > 100:
        story = _target_word_count_story(story, 50, 100)

    if not story.endswith((".", "!", "?")):
        story += "."
    return story

# -------------------------------
# Text-to-speech
# -------------------------------
def text_to_speech(text: str, lang: str = "en") -> str:
    """
    Convert story text to an MP3 file using gTTS.
    Returns temporary file path.
    """
    tts = gTTS(text=text, lang=lang, slow=False)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.set_page_config(page_title="Kids Image Storyteller", page_title="Kids Image Storyteller")
    st.title("Kids Image Storyteller")
    st.markdown("Upload an image and get a short, friendly 50-100 word story with audio.")

    with st.spinner("Loading models (first run may take a few minutes)..."):
        caption_pipe = load_caption_pipeline()
        story_pipe = load_story_pipeline()

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Generate story"):
            try:
                with st.spinner("Generating caption with image-to-text..."):
                    caption = generate_caption(caption_pipe, image)
                    st.info(f"Detected scene: {caption}")

                with st.spinner("Generating story..."):
                    story = generate_story(caption, story_pipe)
                    st.success("Your story:")
                    st.write(story)

                with st.spinner("Converting story to audio..."):
                    audio_path = text_to_speech(story, lang="en")
                    st.audio(audio_path, format="audio/mp3")
                    with open(audio_path, "rb") as f:
                        st.download_button("Download audio", f, file_name="story.mp3", mime="audio/mp3")
                    os.unlink(audio_path)

            except Exception as e:
                st.error(f"Something went wrong: {e}")

if __name__ == "__main__":
    main()
