# modules/pdf_processor.py
import os
import concurrent.futures
import pymupdf
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFProcessor:
    @staticmethod
    def load_pdf(file):
        """PDF file will save by temp.pdf """
        with open("temp.pdf", "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader("temp.pdf")
        return loader.load()

    @staticmethod
    def get_first_page_image(file):
        """Take first page img to preview"""
        with open("temp.pdf", "wb") as f:
            f.write(file.getvalue())
        doc = pymupdf.open("temp.pdf")
        os.makedirs("static", exist_ok=True)
        pix = doc[0].get_pixmap()
        image_path = "static/first_page.png"
        pix.save(image_path)
        return image_path

    @staticmethod
    def chunk_documents(docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_documents(docs)

    @staticmethod
    def process_pdf(file):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_docs = executor.submit(PDFProcessor.load_pdf, file)
            docs = future_docs.result()
            future_chunks = executor.submit(PDFProcessor.chunk_documents, docs)
            documents = future_chunks.result()
        return documents
