# Multi-Modal PDF Processing Demo

This repository contains a multi-modal PDF processing demo built with Streamlit. The demo provides functionalities for:

- **PDF Processing**: Extracting images and text from PDF files.
- **Text Summarization**: Generating a summary of the PDF content using a custom prompt.
- **Visual Question Answering (VQA)**: Answering questions based on a combined image of PDF pages.

All code is organized in an object-oriented manner and the prompt is maintained in a separate module for easier maintenance.

```

PDF-AI-Reader/
├── requirements.txt
├── Dockerfile
└── app
    ├── app.py
    └── modules
        ├── pdf_processor.py
        ├── prompt_doc.py
        ├── text_summarizer.py
        └── visual_qa.py
```

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd project
   ```
2. **Create and activate a virtual environment(optional but recommended):**

   ```python
   # Activate the virtual environment:
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```
3. **Install the dependencies:**

   ```
   pip install -r requirements.txt
   ```

---

## Running the Application

To start the Streamlit app, run:

streamlit run app/app_img.py --server.fileWatcherType none
Once the application is running, your browser will open with the app interface on http://localhost:8501.
Then you can :

Upload a PDF file via the sidebar.
Preview the extracted PDF pages.
Generate a text summary of the PDF content.
Combine PDF images and perform visual question answering.

---

## Running by Dockerfile

If you have Docker, you can running by Dockerfile:

```
docker build -t pdf-ai-reader .
docker run -p 8501:8501 pdf-ai-reader
```
It well open browser in http://localhost:8501.

---

## Additional Information

### Prompt Management:

The text summarization prompt is stored in prompt_doc.py. You can modify the prompt template there without changing the core summarization logic in text_summarizer.py.

### Modules Overview:

pdf_processor.py: Handles saving the PDF file, extracting images, combining images, and extracting text.

visual_qa.py: Manages the Visual Question Answering (VQA) using a pre-trained model.

text_summarizer.py: Uses a text-to-text generation model to summarize PDF content based on a custom prompt.

### Dependencies:

Ensure you have the following packages installed (refer to requirements.txt):

```
Streamlit
PyMuPDF
Pillow
Transformers
langchain_community
```
