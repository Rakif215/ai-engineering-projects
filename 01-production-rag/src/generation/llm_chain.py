import yaml
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class GenerationChain:
    def __init__(self, llm_provider: str = "gemini"):
        self.llm_provider = llm_provider
        self._load_prompts()
        self._init_llm()

    def _load_prompts(self):
        config_path = os.path.join(os.path.dirname(__file__), "prompts", "config.yaml")
        try:
            with open(config_path, "r") as f:
                self.prompts = yaml.safe_load(f)
        except Exception as e:
            print(f"Failed to load prompts: {e}")
            self.prompts = {
                "system_prompt": "You are a helpful assistant.",
                "human_prompt": "Context: {context}\n\nQuestion: {question}"
            }

    def _init_llm(self):
        if self.llm_provider == "groq":
            self.llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
        else:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            
    def _format_docs(self, docs):
        """Formats the retrieved documents into a single string with numbering for citations."""
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            formatted.append(f"[Document {i+1} | Source: {source} | Page: {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def get_chain(self, retriever):
        """
        Builds the RAG chain using the provided retriever and selected LLM.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompts.get("system_prompt")),
            ("human", self.prompts.get("human_prompt"))
        ])

        rag_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def answer(self, question: str, retriever):
        """Generates an answer to the given question based strictly on retrieved context."""
        chain = self.get_chain(retriever)
        return chain.invoke(question)
