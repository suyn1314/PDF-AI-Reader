import os
import streamlit as st
from modules.pdf_processor import PDFProcessor
from modules.visual_qa import VisualQAModel
from modules.text_summarizer import TextSummarizer
from modules.prompt_doc import PDF_SUMMARY_PROMPT


# Disable file watcher in Streamlit for performance
os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"


def main():
    st.set_page_config(layout="wide")
    st.title("🚀 Multi-Modal PDF Processing Demo")
    # Upload & Preview windows
    with st.sidebar:
        st.subheader("📂 Upload PDF File")
        uploaded_file = st.file_uploader("Please upload a PDF file", type="pdf")

        if uploaded_file:
            pdf_processor = PDFProcessor(uploaded_file)
            st.subheader("📜 PDF Page Preview")
            try:
                image_paths = pdf_processor.extract_images()
                for path in image_paths:
                    st.image(path, caption=os.path.basename(path), use_container_width=True)
            except Exception as e:
                st.error("Failed to generate preview images: " + str(e))

    # Main windows
    if uploaded_file:
        text_summarizer = TextSummarizer()
        tab1, tab2 = st.tabs(["📄 PDF Summary", "🖼️ Visual QA"])

        with tab1:
            st.subheader("Generate PDF Summary")
            if st.button("Generate Summary"):
                with st.spinner("Extracting text and generating summary..."):
                    pdf_text = pdf_processor.extract_text()
                    summary = text_summarizer.summarize(pdf_text, PDF_SUMMARY_PROMPT)
                st.write("### PDF Summary:")
                st.write(summary)

        with tab2:
            st.subheader("Visual Question Answering on Combined PDF Image")
            max_pages = st.number_input("Enter number of pages to combine (0 for all)", min_value=0, value=0, step=1)
            if st.button("Combine Images and Ask Question"):
                combined_image = pdf_processor.combine_images(max_pages if max_pages > 0 else None)
                if combined_image:
                    question = st.text_input("Enter your question for the image:")
                    if question:
                        vqa_model = VisualQAModel()
                        answer = vqa_model.get_answer(question, combined_image)
                        st.write("### Answer:")
                        st.write(answer)
                else:
                    st.error("No images available to combine.")
    else:
        st.info("Please upload a PDF file to proceed.")


if __name__ == "__main__":
    main()
