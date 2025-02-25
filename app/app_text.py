# app_text.py
import streamlit as st
from modules.pdf_processor import PDFProcessor
from modules.llm_manager import LLMManager
from modules.qa_builder import QABuilder
from modules.retriever_manager import RetrieverManager


def main():
    st.set_page_config(layout="wide")
    st.title("🚀 Fast RAG-based QA with DeepSeek R1")

    # 側邊欄：上傳 PDF 與預覽
    with st.sidebar:
        st.subheader("📂 Upload PDF & Model Selection")
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        if uploaded_file:
            st.subheader("📜 Preview of Uploaded Document")
            try:
                image_path = PDFProcessor.get_first_page_image(uploaded_file)
                st.image(image_path, caption="First Page Preview", use_column_width=True)
            except Exception as e:
                st.error("Failed to load preview: " + str(e))

    st.subheader("📝 Ask Questions about the Document")
    if uploaded_file:
        with st.spinner("🔄 Processing PDF..."):
            documents = PDFProcessor.process_pdf(uploaded_file)
        retriever = RetrieverManager.create_retriever(documents)
        llm = LLMManager.get_llm()
        qa_chain = QABuilder.build_qa_chain(retriever, llm)
        user_input = st.text_input("Enter your question:")
        if user_input:
            with st.spinner("🤖 Generating response..."):
                response = qa_chain.invoke({"query": user_input})["result"]
                st.write("### 📜 Answer:")
                st.write(response)
    else:
        st.info("📥 Please upload a PDF file to proceed.")


if __name__ == "__main__":
    main()
