import yaml
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

class GenerationChain:
    def __init__(self, llm_provider: str = "gemini"):
        self.llm_provider = llm_provider
        self.chat_history = []
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
        """Formats retrieved chunks into a context string.
        
        Citation labels use the actual filename + page number so that
        one PDF with 3 retrieved chunks does NOT appear as 3 separate 'Documents'.
        Example: [project 1.pdf, p.2]
        """
        formatted = []
        for i, doc in enumerate(docs):
            raw_source = doc.metadata.get("source", "Unknown")
            # Use just the basename so paths don't clutter citations
            source = os.path.basename(raw_source) if raw_source != "Unknown" else "Unknown"
            page = doc.metadata.get("page", "")
            page_label = f", p.{int(page)+1}" if page != "" else ""
            citation = f"[{source}{page_label}]"
            formatted.append(f"{citation}\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def _build_prompt(self):
        """Builds a prompt template that includes conversation history."""
        return ChatPromptTemplate.from_messages([
            ("system", self.prompts.get("system_prompt")),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", self.prompts.get("human_prompt"))
        ])

    def get_chain(self, retriever):
        """Builds the RAG chain with conversation memory support."""
        prompt = self._build_prompt()

        rag_chain = (
            {
                "context": lambda x: self._format_docs(retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"]
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def answer(self, question: str, retriever):
        """Generates an answer and updates conversation history."""
        chain = self.get_chain(retriever)
        response = chain.invoke({
            "question": question,
            "chat_history": self.chat_history
        })
        # Update memory
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response))
        return response

    def stream(self, question: str, retriever):
        """Streams an answer token-by-token and updates conversation history."""
        chain = self.get_chain(retriever)
        full_response = ""
        for chunk in chain.stream({
            "question": question,
            "chat_history": self.chat_history
        }):
            full_response += chunk
            yield chunk
        # Update memory after stream completes
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=full_response))

    def clear_history(self):
        """Resets conversation memory."""
        self.chat_history = []
