"""
RAG (Retrieval Augmented Generation) Engine

This module handles:
- Context retrieval from vector store
- Prompt engineering for different query types
- LLM integration (Gemini, OpenAI, Ollama)
- Response formatting with citations
"""
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from app.core.config import settings
from app.services.vector_store import VectorStore


class RAGEngine:
    """RAG engine for contextual question answering"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM based on provider"""
        provider = settings.LLM_PROVIDER.lower()
        
        if provider == "gemini" and settings.GEMINI_API_KEY:
            # Configure Gemini
            genai.configure(api_key=settings.GEMINI_API_KEY)
            return genai.GenerativeModel(settings.GEMINI_MODEL)
        
        elif provider == "openai" and settings.OPENAI_API_KEY:
            return ChatOpenAI(
                model=settings.OPENAI_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0
            )
        
        elif provider == "ollama":
            return Ollama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL
            )
        
        else:
            raise ValueError(f"Invalid LLM provider: {provider}")
    
    async def query(
        self, 
        question: str, 
        fund_id: Optional[int] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG
        
        Args:
            question: User's question
            fund_id: Optional fund filter
            top_k: Number of context chunks to retrieve
            
        Returns:
            Answer with context and sources
        """
        try:
            # Retrieve relevant context
            filter_metadata = {"fund_id": fund_id} if fund_id else None
            context_results = await self.vector_store.similarity_search(
                query=question,
                k=top_k,
                filter_metadata=filter_metadata
            )
            
            # Format context
            context_text = self._format_context(context_results)
            
            # Generate answer
            answer = await self._generate_answer(question, context_text)
            
            # Format response
            return {
                "answer": answer,
                "sources": self._format_sources(context_results),
                "context_count": len(context_results)
            }
        
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "context_count": 0
            }
    
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved context for the prompt"""
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            score = result.get("score", 0)
            context_parts.append(
                f"[Context {i}] (Relevance: {score:.2f})\n{content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _format_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sources for citation"""
        sources = []
        for result in results:
            sources.append({
                "document_id": result.get("document_id"),
                "fund_id": result.get("fund_id"),
                "score": round(result.get("score", 0), 3),
                "preview": result.get("content", "")[:200] + "..."
            })
        return sources
    
    async def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM"""
        
        # Build prompt
        prompt = self._build_prompt(question, context)
        
        # Generate response based on provider
        if isinstance(self.llm, genai.GenerativeModel):
            # Gemini
            response = self.llm.generate_content(prompt)
            return response.text
        
        elif hasattr(self.llm, 'invoke'):
            # LangChain compatible (OpenAI, Ollama)
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        
        else:
            # Fallback
            return "LLM not properly configured"
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for LLM"""
        
        prompt = f"""You are a financial analyst assistant specializing in fund performance analysis.

Your task is to answer questions about fund performance metrics, investment strategies, and financial documents.

**Context Information:**
{context}

**Question:**
{question}

**Instructions:**
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information, clearly state that
3. Be concise and precise
4. For definitions, provide clear explanations
5. For calculations, explain the formula and components
6. Include relevant citations from the context

**Answer:**"""
        
        return prompt
    
    async def answer_definition_query(self, term: str, fund_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Answer definition queries (e.g., "What is DPI?")
        
        Args:
            term: Term to define
            fund_id: Optional fund filter
            
        Returns:
            Definition with context
        """
        question = f"What is {term}? Provide a clear definition and explanation."
        return await self.query(question, fund_id, top_k=3)
    
    async def answer_data_query(self, query: str, fund_id: int) -> Dict[str, Any]:
        """
        Answer data retrieval queries (e.g., "Show me all capital calls in 2024")
        
        Args:
            query: Data query
            fund_id: Fund ID
            
        Returns:
            Answer with retrieved data
        """
        return await self.query(query, fund_id, top_k=5)
    
    async def explain_calculation(
        self, 
        metric: str, 
        fund_id: int,
        calculation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Explain a calculation result with context
        
        Args:
            metric: Metric name (e.g., "DPI", "IRR")
            fund_id: Fund ID
            calculation_result: Calculation breakdown
            
        Returns:
            Explanation with context
        """
        # Build query with calculation data
        question = f"""Explain the {metric} calculation for this fund.

Calculation Result:
{calculation_result}

Provide a clear explanation of:
1. What this metric means
2. How it was calculated
3. What the result indicates about fund performance
"""
        
        return await self.query(question, fund_id, top_k=3)
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings"""
        try:
            from sqlalchemy import text
            
            stats_query = text("""
                SELECT 
                    COUNT(*) as total_embeddings,
                    COUNT(DISTINCT document_id) as unique_documents,
                    COUNT(DISTINCT fund_id) as unique_funds
                FROM document_embeddings
            """)
            
            result = self.vector_store.db.execute(stats_query).fetchone()
            
            return {
                "total_embeddings": result[0] if result else 0,
                "unique_documents": result[1] if result else 0,
                "unique_funds": result[2] if result else 0
            }
        except Exception as e:
            return {
                "error": str(e),
                "total_embeddings": 0,
                "unique_documents": 0,
                "unique_funds": 0
            }
