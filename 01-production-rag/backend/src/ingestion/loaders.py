from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, WebBaseLoader
import os

class DocumentIngestor:
    def __init__(self):
        pass

    def load_pdf(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at {file_path}")
        loader = PyPDFLoader(file_path)
        return loader.load()

    def load_markdown(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Markdown file not found at {file_path}")
        loader = UnstructuredMarkdownLoader(file_path)
        return loader.load()

    def load_web_page(self, url: str):
        loader = WebBaseLoader(url)
        return loader.load()

    def ingest_directory(self, dir_path: str):
        # Helper to ingest all supported files in a directory
        documents = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.pdf'):
                    documents.extend(self.load_pdf(file_path))
                elif file.endswith('.md'):
                    documents.extend(self.load_markdown(file_path))
        return documents

