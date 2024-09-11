# Visual Question Answering App
This Visual Question Answering (VQA) App allows users to upload an image and ask a question related to it. The app uses a fine-tuned transformer model from Hugging Face to provide an answer based on the image content.

Features
Image Upload: Users can upload an image in jpg, jpeg, or png formats.
Question Input: Users can input a text-based question related to the uploaded image.
Answer Generation: The app will predict an answer based on the image and the question using the ViltForQuestionAnswering model.
Easy to Use: Simple interface powered by Streamlit, with an intuitive workflow for uploading images and asking questions.
Technologies Used
Streamlit: For building the web interface.
Transformers: From Hugging Face for loading the pre-trained VQA model.
PIL: For handling images.
Python: The core programming language used
