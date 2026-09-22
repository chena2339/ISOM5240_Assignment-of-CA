import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS
import tempfile
import os

# -------------------------------
# Load Models (cached to avoid reloading)
# -------------------------------
@st.cache_resource
def load_caption_model():
    """Load the BLIP image captioning pipeline (auto-detect task)."""
    # No explicit task parameter – let Hugging Face infer the correct pipeline
    return pipeline(model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_model():
    """Load a text generation pipeline (GPT-2) for story expansion."""
    return pipeline("text-generation", model="gpt2")

# -------------------------------
# Core Functions
# -------------------------------
def generate_caption(image: Image.Image, caption_pipe) -> str:
    """
    Generate a caption for the given image using the captioning pipeline.
    Ensures the image is in RGB mode before passing to the model.
    Returns a string description.
    """
    result = caption_pipe(image.convert("RGB"))
    # BLIP returns a list of dicts; take the first generated text
    caption = result[0]["generated_text"]
    return caption.strip()

def generate_story(caption: str, story_pipe, max_length=150) -> str:
    """
    Expand the caption into a short story (50-100 words).
    Uses GPT-2 with a prompt tailored for children.
    """
    prompt = (
        f"Once upon a time, there was a scene: {caption}. "
        "Tell a short and fun story for kids about what happens next."
    )
    result = story_pipe(
        prompt,
        max_length=max_length,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        truncation=True
    )
    full_text = result[0]["generated_text"]
    # Remove the original prompt from the output
    story = full_text[len(prompt):].strip()
    # Ensure story ends properly; keep roughly 3 sentences
    sentences = story.split(".")
    short_story = ".".join(sentences[:3]) + "."
    # Word count check: aim for 50-100 words
    words = short_story.split()
    if len(words) > 110:
        short_story = " ".join(words[:100]) + "..."
    elif len(words) < 20:
        # Fallback: try again with higher temperature
        result2 = story_pipe(
            prompt,
            max_length=200,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            truncation=True
        )
        short_story = result2[0]["generated_text"][len(prompt):].strip()
        short_story = ".".join(short_story.split(".")[:3]) + "."
    return short_story

def text_to_speech(text: str) -> str:
    """
    Convert text to speech using gTTS and save to a temporary MP3 file.
    Returns the path to the audio file.
    """
    tts = gTTS(text=text, lang="en")
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    return tmp_file.name

# -------------------------------
# Streamlit App UI
# -------------------------------
def main():
    st.set_page_config(page_title="Kids Storyteller 🧸", layout="centered")
    st.title("📖 Storytelling App for Kids (3-10 years)")
    st.markdown("Upload an image and I'll tell you a magical story!")

    # Load models once
    with st.spinner("Loading AI models... this may take a minute ⏳"):
        caption_pipe = load_caption_model()
        story_pipe = load_story_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Your picture", use_container_width=True)

        # Generate story button
        if st.button("✨ Tell me a story!"):
            with st.spinner("Looking at your picture..."):
                caption = generate_caption(image, caption_pipe)
                st.info(f"I see: {caption}")

            with st.spinner("Creating a story..."):
                story = generate_story(caption, story_pipe)
                st.success("Here's your story!")
                st.write(story)

            with st.spinner("Converting to speech..."):
                audio_path = text_to_speech(story)
                st.audio(audio_path, format="audio/mp3")
                st.download_button(
                    label="Download Audio",
                    data=open(audio_path, "rb"),
                    file_name="story.mp3",
                    mime="audio/mp3"
                )
                # Clean up temporary file
                os.unlink(audio_path)

if __name__ == "__main__":
    main()
