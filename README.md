# Visual Question Answering App
The Visual Question Answering (VQA) App allows users to upload an image and ask a related question, with the app providing an AI-generated answer based on the image content. It utilizes a pre-trained model from Hugging Face, specifically the ViltForQuestionAnswering model fine-tuned for visual question answering tasks. The app is built using Streamlit for the web interface, and it handles image processing through PIL (Python Imaging Library). To use the app, users upload an image in jpg, jpeg, or png format, input a question, and the app predicts the answer using the ViltForQuestionAnswering model.

To run the app locally, users can clone the repository, set up a virtual environment, and install the required dependencies, including Streamlit, Transformers, and Pillow. The app can be started by running streamlit run app.py. Once running, the interface allows users to upload an image, type a question, and click a button to receive the answer. The app is simple to use and provides answers based on the content of the uploaded image and the user’s query.

Install the required dependencies:pip install -r requirements.txt


MODEL USED IS dandelin/vilt-b32-finetuned-vqa

