from transformers import pipeline
import streamlit as st


@st.cache_resource
def get_vqa_pipeline():
    """
    Initialize and cache the visual-question-answering pipeline.
    Model: Salesforce/blip2-flan-t5-xl
    """
    return pipeline("visual-question-answering", model="Salesforce/blip2-flan-t5-xl")


class VisualQAModel:
    """
    Class for handling Visual Question Answering (VQA) tasks.

    Attributes:
        pipeline: The VQA pipeline instance.
    """

    def __init__(self):
        self.pipeline = get_vqa_pipeline()

    def get_answer(self, question, image):
        """
        Generate an answer for the given question using the provided image.

        Args:
            question: The question as a string.
            image: A PIL Image object.

        Returns:
            The generated answer as a string.
        """
        result = self.pipeline(image=image, question=question)
        if isinstance(result, list) and len(result) > 0:
            return result[0]['answer']
        return "Unable to generate answer."
