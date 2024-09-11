import streamlit as st
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, ViltForQuestionAnswering

# Set the page layout to wide mode
st.set_page_config(layout="wide")


# Load the processor and model from Hugging Face
processor = AutoProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Function to get an answer from the model using the image and text
def get_answer(image, text):
    try:
        # Convert image bytes to a PIL Image and make sure it's RGB
        img = Image.open(BytesIO(image)).convert("RGB")
        st.write(f"Image size: {img.size}")  # Debugging: Check image size

        # Process the image and text using the processor
        encoding = processor(img, text, return_tensors="pt")

        # Pass the inputs to the model and get the outputs
        outputs = model(**encoding)
        logits = outputs.logits
        #This is the list of scores (logits) for each answer.
        #The model’s output contains the logits for all possible answers.

        # Get the index of the highest scoring answer
        idx = logits.argmax(-1).item()#the most confident prediction

        # Retrieve the answer label from the model's configuration
        answer = model.config.id2label[idx]

        return answer

    except Exception as e:
        # If any error occurs, return the error message
        return str(e)

# Set up the title and description for the app
st.title("Visual Question Answering")
st.write("Upload an image and enter a question to get an answer.")

# Create two columns: one for image upload and one for question input
col1, col2 = st.columns(2)

# Left Column: Image Upload Section
with col1:
    # Allow the user to upload an image file
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Display debugging information about the uploaded file
        st.write(f"Uploaded file type: {type(uploaded_file)}")  # Debugging: Check file type
        st.write(f"Uploaded file size: {uploaded_file.size} bytes")  # Debugging: Check file size

        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)  # Display the image in the column

# Right Column: Question Input Section
with col2:
    # Allow the user to input a question related to the uploaded image
    question = st.text_input("Question")

    # Check if both an image and a question have been provided
    if uploaded_file and question:
        # When the "Ask Question" button is clicked, process the image and question
        if st.button("Ask Question"):
            try:
                # Convert the uploaded image to bytes
                image_byte_array = BytesIO()
                image.save(image_byte_array, format='jpeg')
                image_bytes = image_byte_array.getvalue()

                # Call the function to get an answer from the model
                answer = get_answer(image_bytes, question)

                # Display the answer in a success box
                st.success("Answer: " + answer)
            except Exception as e:
                # If any error occurs, display an error message
                st.error(f"Error processing the image or question: {e}")
