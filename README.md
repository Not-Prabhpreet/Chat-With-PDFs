# Chat with Multiple PDFs 📚

Welcome to the Chat with Multiple PDFs project! This application allows you to upload multiple PDF documents, ask questions about their content, and get detailed answers. The app uses advanced language models to process and understand the text from the PDFs and provide accurate responses to your queries.

## Live Demo

Check out the live version of the project [here](https://chat-with-pdfs-dlalukfv6fe3adcp88uagc.streamlit.app/).

## Features

- Upload multiple PDF documents.
- Extract and process text from PDFs.
- Ask questions about the content of the PDFs.
- Get detailed and accurate answers.

## How to Use

1. **Upload Your PDFs**: Use the sidebar to upload one or more PDF documents.
2. **Ask Questions**: Enter your question in the text input field.
3. **Get Answers**: The app will process your question and provide a detailed response based on the content of the uploaded PDFs.

## Installation

To run this project locally, follow these steps:
<pre> 
1. Clone this repository:
   ```bash
   git clone https://github.com/Not-Prabhpreet/Chat-With-PDFs.git
  
2. Navigate to the project directory:
   cd Chat-With-PDFs
  
3. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  
4. Install the required dependencies:
   pip install -r requirements.txt
  
5. Set up the environment variables. Create a .env file in the project root and add your Google API key:
   GOOGLE_API_KEY=your_google_api_key
  
6. Run the Streamlit App:
   streamlit run app.py
</pre>


   
## Tech Stack Used:
<pre>
-Python
-Streamlit
-PyPDF2
-LangChain
-FAISS
-Google Generative AI
</pre>




