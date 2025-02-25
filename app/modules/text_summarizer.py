from transformers import pipeline
import streamlit as st


@st.cache_resource
def get_text_summarizer_pipeline():
    """
    Initialize and cache the text summarization pipeline.
    Model: google/flan-t5-xl
    """
    return pipeline("text2text-generation", model="google/flan-t5-xl")


class TextSummarizer:
    """
    Class for summarizing text content.

    Attributes:
        pipeline: The text-to-text generation pipeline instance.
    """

    def __init__(self):
        self.pipeline = get_text_summarizer_pipeline()

    def summarize(self, pdf_text, prompt_text):
        """
        Generate a summary for the provided PDF text using the prompt from prompt_doc.py.

        Args:
            pdf_text: The full text content of the PDF.
            prompt_text: The prompt text of the document.

        Returns:
            The generated summary as a string.
        """
        # Format the prompt with the PDF text
        prompt = prompt_text.format(content=pdf_text)
        summary_result = self.pipeline(prompt, max_length=300, truncation=True)
        return summary_result[0]['generated_text']
