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
        model="google/flan-t5-large"
    )

    def clean_story_output(output_text):
        output_text = output_text.strip()
        output_text = output_text.replace("Story:", "").strip()
        output_text = output_text.replace("Expanded story:", "").strip()
        output_text = output_text.replace("Continuation:", "").strip()
        output_text = output_text.replace("New continuation:", "").strip()
        output_text = output_text.strip('"').strip("'").strip()
        return output_text

    # First generation: create a story draft
    prompt = (
        "You are a children's story writer. "
        "Write the beginning of one warm children's story in simple English. "
        "The story is based on this image description: "
        + text
        + ". "
        "Do not repeat the image description. "
        "Do not write instructions, lists, or titles. "
        "Only output story text."
    )

    story_results = story_pipe(
        prompt,
        max_new_tokens=120,
        min_new_tokens=50,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.8,
        no_repeat_ngram_size=3
    )

    story_text = clean_story_output(story_results[0]["generated_text"])

    # Continue the story until it reaches at least 50 words
    expand_round = 0

    while len(story_text.split()) < 50 and expand_round < 5:
        expand_prompt = (
            "Continue the children's story below by adding two new simple sentences. "
            "The continuation must be warm, positive, imaginative, and suitable for children aged 3 to 10. "
            "Do not repeat any previous sentence. "
            "Do not restart the story. "
            "Do not write instructions, lists, or titles. "
            "Only output the new continuation sentences. "
            "Image description: "
            + text
            + ". "
            "Current story: "
            + story_text
        )

        expand_results = story_pipe(
            expand_prompt,
            max_new_tokens=90,
            min_new_tokens=35,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3
        )

        continuation = clean_story_output(expand_results[0]["generated_text"])

        # If the model repeats the current story, remove the repeated part
        if story_text.lower() in continuation.lower():
            continuation = continuation.lower().replace(story_text.lower(), "").strip()

        # Append only useful continuation
        if len(continuation.split()) >= 6 and continuation.lower() not in story_text.lower():
            story_text = story_text.rstrip(". ") + ". " + continuation.strip()

        expand_round += 1

    # If the story is longer than 100 words, keep only the first 100 words
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
