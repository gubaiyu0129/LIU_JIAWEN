# Program title: Image to Audio Storytelling App

# Import part
import streamlit as st
from transformers import pipeline


# Function part

# img2text
def img2text(url):
    image_to_text_model = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )

    text = image_to_text_model(url)[0]["generated_text"]

    return text


# text2story
def text2story(text):
    story_pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

    prompt = (
        "Write only one short children's story in simple English. "
        "The story must be 50 to 100 words. "
        "It must be warm, positive, and suitable for children aged 3 to 10. "
        "Do not repeat the instructions. "
        "Do not include scary, violent, or bad behavior. "
        "Use this image description as the story idea: "
        + text
    )

    story_results = story_pipe(
        prompt,
        max_new_tokens=130,
        do_sample=True,
        temperature=0.7
    )

    story_text = story_results[0]["generated_text"].strip()

    # Basic check: if the model repeats the prompt or gives an unsuitable short answer,
    # create a safe backup story based on the image description.
    bad_words = [
        "requirement",
        "instruction",
        "should be",
        "thief",
        "violent",
        "scary"
    ]

    if len(story_text.split()) < 40 or any(word in story_text.lower() for word in bad_words):
        story_text = (
            f"One sunny morning, a kind character from the picture, described as {text}, "
            f"found a little bird looking for its way home. The character smiled gently "
            f"and followed the bird through a quiet garden. Along the way, they saw bright "
            f"flowers, tiny butterflies, and a sparkling pond. At last, the bird found its "
            f"cozy nest. Everyone felt happy, and they became good friends."
        )

    # Limit the story to about 100 words
    words = story_text.split()
    if len(words) > 100:
        story_text = " ".join(words[:100]) + "."

    return story_text


# text2audio
def text2audio(story_text):
    audio_pipe = pipeline(
        "text-to-audio",
        model="Matthijs/mms-tts-eng"
    )

    audio_data = audio_pipe(story_text)

    return audio_data


# Main part
def main():
    st.set_page_config(
        page_title="Image to Audio Story",
        page_icon="🦜",
        layout="centered"
    )

    # CSS style
    st.markdown(
        """
        <style>
        .main-title {
            text-align: center;
            font-size: 42px;
            font-weight: 800;
            color: #3B3B58;
            margin-bottom: 10px;
        }

        .sub-title {
            text-align: center;
            font-size: 18px;
            color: #666666;
            margin-bottom: 30px;
        }

        .story-card {
            background-color: #FFF8E7;
            padding: 24px;
            border-radius: 18px;
            border: 1px solid #F1DCA7;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.06);
            margin-top: 15px;
            margin-bottom: 20px;
        }

        .scenario-card {
            background-color: #EEF6FF;
            padding: 18px;
            border-radius: 16px;
            border: 1px solid #CFE6FF;
            margin-top: 15px;
            margin-bottom: 15px;
        }

        .audio-card {
            background-color: #F1FFF4;
            padding: 18px;
            border-radius: 16px;
            border: 1px solid #CDEFD6;
            margin-top: 15px;
            margin-bottom: 20px;
        }

        .small-note {
            color: #777777;
            font-size: 14px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.title("📚 App Info")
        st.write("This app creates a short audio story from an uploaded image.")
        st.write("Designed for children aged **3–10**.")
        st.write("Main stages:")
        st.write("1. Image Captioning")
        st.write("2. Story Generation")
        st.write("3. Text-to-Audio")
        st.divider()
        st.write("Built with:")
        st.write("✅ Streamlit")
        st.write("✅ Hugging Face Pipeline")

    # Main title
    st.markdown(
        '<div class="main-title">🦜 Image to Audio Story</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Upload a picture and let the app create a warm children\'s story with audio.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "📤 Select an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Save file locally
        bytes_data = uploaded_file.getvalue()

        file_extension = uploaded_file.name.split(".")[-1]
        image_path = "uploaded_image." + file_extension

        with open(image_path, "wb") as file:
            file.write(bytes_data)

        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_column_width=True
        )

        st.markdown(
            '<p class="small-note">Click the button below to generate your story.</p>',
            unsafe_allow_html=True
        )

        generate_button = st.button(
            "✨ Generate My Story",
            type="primary",
            use_container_width=True
        )

        if generate_button:

            try:
                # Stage 1: Image to Text
                with st.spinner("🔍 Looking at the image..."):
                    scenario = img2text(image_path)

                st.markdown(
                    f"""
                    <div class="scenario-card">
                    <h4>🖼️ Image Scenario</h4>
                    <p>{scenario}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Stage 2: Text to Story
                with st.spinner("✍️ Writing a short children's story..."):
                    story = text2story(scenario)

                st.markdown(
                    f"""
                    <div class="story-card">
                    <h4>📖 Generated Story</h4>
                    <p>{story}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Stage 3: Story to Audio
                with st.spinner("🎧 Creating audio for the story..."):
                    audio_data = text2audio(story)

                audio_array = audio_data["audio"]
                sample_rate = audio_data["sampling_rate"]

                st.markdown(
                    """
                    <div class="audio-card">
                    <h4>🔊 Listen to the Story</h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.audio(audio_array, sample_rate=sample_rate)

                st.success("Done! Your image story is ready. 🌟")

            except Exception as e:
                st.error("An error occurred while running the app.")
                st.exception(e)

    else:
        st.info("Please upload an image to start the story generation.")


if __name__ == "__main__":
    main()
