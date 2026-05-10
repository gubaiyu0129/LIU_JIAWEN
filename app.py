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
        page_title="Your Image to Audio Story",
        page_icon="🦜"
    )

    st.header("Turn Your Image into an Audio Story")

    st.write(
        "Upload an image, and this app will generate a short children's story and turn it into audio."
    )

    uploaded_file = st.file_uploader(
        "Select an Image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Save file locally
        bytes_data = uploaded_file.getvalue()

        file_extension = uploaded_file.name.split(".")[-1]
        image_path = "uploaded_image." + file_extension

        with open(image_path, "wb") as file:
            file.write(bytes_data)

        # Display uploaded image
        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_column_width=True
        )

        if st.button("Generate Story"):

            try:
                # Stage 1: Image to Text
                st.text("Processing img2text...")
                scenario = img2text(image_path)
                st.write(f"**Scenario:** {scenario}")

                # Stage 2: Text to Story
                st.text("Generating a story...")
                story = text2story(scenario)
                st.write(f"**Story:** {story}")

                # Stage 3: Story to Audio
                st.text("Generating audio data...")
                audio_data = text2audio(story)

                # Play audio
                audio_array = audio_data["audio"]
                sample_rate = audio_data["sampling_rate"]

                st.audio(audio_array, sample_rate=sample_rate)

            except Exception as e:
                st.error("An error occurred while running the app.")
                st.exception(e)


if __name__ == "__main__":
    main()
