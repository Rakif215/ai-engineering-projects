from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        # We target 500-800 token chunks. Using RecursiveCharacterTextSplitter 
        # as a starting point, which splits by characters, typically 1 token ~ 4 characters.
        # But we can approximate tokens using a tokenizer if needed later. 
        # For now, we'll use character limits roughly equivalent to tokens.
        # 800 tokens * 4 chars/token ~ 3200 characters.
        self.char_chunk_size = chunk_size * 4
        self.char_chunk_overlap = chunk_overlap * 4
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.char_chunk_size,
            chunk_overlap=self.char_chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_documents(self, documents):
        """
        Splits a list of langchain Documents into smaller chunks.
        """
        return self.text_splitter.split_documents(documents)
