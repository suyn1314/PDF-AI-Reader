# modules/qa_builder.py
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA


class QABuilder:
    @staticmethod
    def build_prompt():
        prompt = """
        1. Use the following pieces of context to answer the question at the end.
        2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.
        3. Keep the answer crisp and limited to 3-4 sentences.

        Context: {context}

        Question: {question}

        Helpful Answer:"""
        return PromptTemplate.from_template(prompt)

    @staticmethod
    def build_qa_chain(retriever, llm):
        prompt = QABuilder.build_prompt()
        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="Context:\ncontent:{page_content}\nsource:{source}",
        )
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
        )
        return RetrievalQA(
            combine_documents_chain=combine_documents_chain,
            retriever=retriever,
            return_source_documents=True,
        )
