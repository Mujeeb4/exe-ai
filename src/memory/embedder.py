"""
Embedding logic for converting code chunks to vectors.
Supports multiple providers: Google (genai) and OpenAI.
"""

from typing import List, Optional
import openai


class Embedder:
    """Handles text-to-vector embedding using multiple providers."""
    
    # Embedding dimensions for each model
    EMBEDDING_DIMS = {
        "text-embedding-004": 768,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "text-embedding-004", 
        provider: str = "google"
    ):
        """
        Initialize embedder with specified provider.
        
        Args:
            api_key: API key for the provider
            model: Embedding model name
            provider: Provider name ("google" or "openai")
        """
        self.model = model
        self.provider = provider
        self.api_key = api_key
        
        if provider == "openai":
            self.client = openai.OpenAI(api_key=api_key)
        elif provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.genai = genai
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension for the current model."""
        return self.EMBEDDING_DIMS.get(self.model, 768)
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if self.provider == "openai":
            return self._embed_openai(texts)
        else:
            return self._embed_google(texts)
    
    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def _embed_google(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Google Generative AI."""
        embeddings = []
        for text in texts:
            result = self.genai.embed_content(
                model=f"models/{self.model}",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        return embeddings
    
    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        return self.embed([text])[0]

