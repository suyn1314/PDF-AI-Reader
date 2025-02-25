import streamlit as st
import concurrent.futures
import os
os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"
import fitz  # PyMuPDF
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from transformers import pipeline


###############################
# MULTIMODAL IMAGE QA SECTION #
###############################

@st.cache_resource
def get_vqa_pipeline():
    """
    Initialize the visual-question-answering pipeline using BLIP-2.
    (Using Salesforce/blip2-flan-t5-xl as an example of a multimodal model.)
    """
    return pipeline("visual-question-answering", model="Salesforce/blip2-flan-t5-xl")


def save_pdf_temp(file):
    """Save the uploaded PDF as a temporary file."""
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(file.getvalue())
    return temp_path


def get_pdf_images(file):
    """Convert each page of the PDF into images and return a list of image file paths."""
    temp_pdf = save_pdf_temp(file)
    doc = fitz.open(temp_pdf)
    image_paths = []
    os.makedirs("static", exist_ok=True)
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap()
        img_path = os.path.join("static", f"page_{i + 1}.png")
        pix.save(img_path)
        image_paths.append(img_path)
    return image_paths


def combine_images(image_paths, max_pages=None):
    """
    Vertically combine multiple images into one.
    If max_pages is set, only the first max_pages images are combined.
    """
    imgs = [Image.open(path) for path in image_paths[:max_pages] if os.path.exists(path)]
    if not imgs:
        return None
    widths, heights = zip(*(img.size for img in imgs))
    max_width = max(widths)
    total_height = sum(heights)
    combined = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for img in imgs:
        combined.paste(img, (0, y_offset))
        y_offset += img.size[1]
    return combined


def get_multimodal_answer(question, combined_image):
    """
    Call the multimodal model to obtain an answer for the given question.
    The model accepts both an image and a text prompt.
    """
    vqa = get_vqa_pipeline()
    result = vqa(image=combined_image, question=question)
    return result[0]['answer'] if isinstance(result, list) and len(result) > 0 else "Unable to generate answer."


####################################
# TEXT SUMMARIZATION WITH PROMPTING #
####################################

@st.cache_resource
def get_text_summarizer():
    """
    Initialize a text-to-text generation pipeline.
    Here we use "google/flan-t5-xl" which can follow custom instructions.
    """
    return pipeline("text2text-generation", model="google/flan-t5-xl")


def load_pdf_text(file):
    """
    Load the PDF text using PyPDFLoader.
    This function extracts text from all pages and concatenates them.
    """
    temp_path = save_pdf_temp(file)
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    full_text = "\n".join(doc.page_content for doc in docs)
    return full_text


def get_pdf_summary(pdf_text, summarizer):
    """
    The prompt pdf summary must include:
      1. A brief description of the overall content.
      2. Detailed descriptions of any diagrams/charts present.
      3. Abstract summaries of any tables included.
    """
    prompt = (
        "You are a professional document summarizer. Please summarize the following PDF content. "
        "Your summary should include detailed descriptions of any diagrams or charts as well as abstract summaries of any tables present. "
        "Ensure your summary is clear, concise, and covers the key points of the document.\n\n"
        "PDF Content:\n{content}\n\nSummary:"
    ).format(content=pdf_text)
    summary = summarizer(prompt, max_length=300, truncation=True)[0]['generated_text']
    return summary


######################
# STREAMLIT APP MAIN #
######################

def main():
    st.set_page_config(layout="wide")
    st.title("🚀 Multi-Modal PDF Processing Demo")

    # Upload file and initialize pipelines once
    with st.sidebar:
        st.subheader("📂 Upload PDF File")
        uploaded_file = st.file_uploader("Please upload a PDF file", type="pdf")
        if uploaded_file:
            st.subheader("📜 PDF Page Preview")
            try:
                image_paths = get_pdf_images(uploaded_file)
                for path in image_paths:
                    st.image(path, caption=os.path.basename(path), use_container_width=True)
            except Exception as e:
                st.error("Failed to generate preview images: " + str(e))

    # Initialize the text summarizer once if file exists
    summarizer = None
    if uploaded_file:
        summarizer = get_text_summarizer()

    tab1, tab2 = st.tabs(["💬 Q&A with Images", "📄 PDF Summary"])

    with tab1:
        st.subheader("Ask Questions about the PDF (Image-based)")
        if uploaded_file:
            user_question = st.text_input("Enter your question:")
            if user_question:
                with st.spinner("Generating answer..."):
                    image_paths = get_pdf_images(uploaded_file)
                    combined_img = combine_images(image_paths, max_pages=3)
                    if combined_img is None:
                        st.error("Failed to combine images.")
                        return
                    st.image(combined_img, caption="Combined Image Preview", use_container_width=True)
                    answer = get_multimodal_answer(user_question, combined_img)
                st.write("### Answer:")
                st.write(answer)
        else:
            st.info("Please upload a PDF file to proceed.")

    with tab2:
        st.subheader("Generate PDF Summary (Text-based)")
        if uploaded_file:
            if st.button("Generate Summary"):
                with st.spinner("Extracting text and generating summary..."):
                    pdf_text = load_pdf_text(uploaded_file)
                    # Use the already initialized summarizer
                    summary = get_pdf_summary(pdf_text, summarizer)
                st.write("### PDF Summary:")
                st.write(summary)
        else:
            st.info("Please upload a PDF file to proceed.")


if __name__ == "__main__":
    main()
