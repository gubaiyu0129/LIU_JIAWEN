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
        "text-generation",
        model="pranavpsv/genre-story-generator-v2"
    )

    story_prompt = f"""
    Write a short and cheerful story for children aged 3 to 10.

    The story should be based on this image description:
    {text}

    Requirements:
    - Use simple English.
    - Make the story warm and imaginative.
    - The story should be about 50 to 100 words.
    - The story should have a happy ending.
    """

    story_results = story_pipe(
        story_prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.8,
        top_p=0.95
    )

    story_text = story_results[0]["generated_text"]

    # Remove the prompt part if the model includes it in the output
    if story_prompt in story_text:
        story_text = story_text.replace(story_prompt, "").strip()

    return story_text

# text2story
def text2story(text):
    story_pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

    story_prompt = f"""
    Write a short, warm, and imaginative story for children aged 3 to 10.

    Image description:
    {text}

    Requirements:
    The story should be 50 to 100 words.
    Use simple English.
    Make the story positive and child-friendly.
    Avoid scary or violent details.
    Include a happy ending.
    """

    story_results = story_pipe(
        story_prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.8
    )

    story_text = story_results[0]["generated_text"].strip()

    # Keep the story within about 100 words
    words = story_text.split()
    if len(words) > 100:
        story_text = " ".join(words[:100]) + "."

    return story_text

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
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Generate Story"):

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


if __name__ == "__main__":
    main()
