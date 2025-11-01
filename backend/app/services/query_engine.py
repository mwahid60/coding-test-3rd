"""
Query engine service for RAG-based question answering
"""
from typing import Dict, Any, List, Optional
import time
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
from app.core.logger import get_logger
from app.services.vector_store import VectorStore
from app.services.metrics_calculator import MetricsCalculator
from sqlalchemy.orm import Session

logger = get_logger(__name__)


class QueryEngine:
    """RAG-based query engine for fund analysis"""
    
    def __init__(self, db: Session):
        logger.info("🔧 Initializing QueryEngine")
        self.db = db
        self.vector_store = VectorStore()
        self.metrics_calculator = MetricsCalculator(db)
        self.llm = self._initialize_llm()
        logger.info("✅ QueryEngine initialized successfully")
    
    def _initialize_llm(self):
        """Initialize LLM"""
        logger.info("🤖 Initializing LLM")
        if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "":
            genai.configure(api_key=settings.GEMINI_API_KEY)
            logger.info(f"✅ Using Gemini model: {settings.GEMINI_MODEL}")
            return ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                temperature=0,
                google_api_key=settings.GEMINI_API_KEY
            )
        else:
            logger.error("❌ GEMINI_API_KEY not configured")
            raise ValueError("GEMINI_API_KEY not configured. Please set it in .env file")
    
    async def process_query(
        self, 
        query: str, 
        fund_id: Optional[int] = None,
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query using RAG
        
        Args:
            query: User question
            fund_id: Optional fund ID for context
            conversation_history: Previous conversation messages
            
        Returns:
            Response with answer, sources, and metrics
        """
        start_time = time.time()
        logger.info(f"▶️  Processing query: '{query[:50]}...' (fund_id={fund_id})")
        
        # Step 1: Classify query intent
        intent = await self._classify_intent(query)
        logger.info(f"🎯 Query intent classified as: {intent}")
        
        # Step 2: Retrieve relevant context from vector store
        filter_metadata = {"fund_id": fund_id} if fund_id else None
        logger.info(f"🔍 Searching vector store (k={settings.TOP_K_RESULTS})")
        relevant_docs = await self.vector_store.similarity_search(
            query=query,
            k=settings.TOP_K_RESULTS,
            filter_metadata=filter_metadata
        )
        logger.info(f"📄 Found {len(relevant_docs)} relevant documents")
        
        # Step 3: Calculate metrics if needed
        metrics = None
        if intent == "calculation" and fund_id:
            logger.info(f"📊 Calculating metrics for fund_id={fund_id}")
            metrics = self.metrics_calculator.calculate_all_metrics(fund_id)
        
        # Step 4: Generate response using LLM
        logger.info("🤖 Generating LLM response")
        answer = await self._generate_response(
            query=query,
            context=relevant_docs,
            metrics=metrics,
            conversation_history=conversation_history or []
        )
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Query processed successfully in {processing_time:.2f}s")
        
        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc["content"],
                    "metadata": {
                        k: v for k, v in doc.items() 
                        if k not in ["content", "score"]
                    },
                    "score": doc.get("score")
                }
                for doc in relevant_docs
            ],
            "metrics": metrics,
            "processing_time": round(processing_time, 2)
        }
    
    async def _classify_intent(self, query: str) -> str:
        """
        Classify query intent
        
        Returns:
            'calculation', 'definition', 'retrieval', or 'general'
        """
        logger.debug(f"Classifying intent for query: {query[:50]}...")
        query_lower = query.lower()
        
        # Calculation keywords
        calc_keywords = [
            "calculate", "what is the", "current", "dpi", "irr", "tvpi", 
            "rvpi", "pic", "paid-in capital", "return", "performance"
        ]
        if any(keyword in query_lower for keyword in calc_keywords):
            return "calculation"
        
        # Definition keywords
        def_keywords = [
            "what does", "mean", "define", "explain", "definition", 
            "what is a", "what are"
        ]
        if any(keyword in query_lower for keyword in def_keywords):
            return "definition"
        
        # Retrieval keywords
        ret_keywords = [
            "show me", "list", "all", "find", "search", "when", 
            "how many", "which"
        ]
        if any(keyword in query_lower for keyword in ret_keywords):
            return "retrieval"
        
        return "general"
    
    async def _generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]],
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """Generate response using LLM"""
        logger.debug(f"Generating response with {len(context)} context documents")
        
        # Build context string
        context_str = "\n\n".join([
            f"[Source {i+1}]\n{doc['content']}"
            for i, doc in enumerate(context[:3])  # Use top 3 sources
        ])
        logger.debug(f"Built context string with {len(context[:3])} sources")
        
        # Build metrics string
        metrics_str = ""
        if metrics:
            metrics_str = "\n\nAvailable Metrics:\n"
            for key, value in metrics.items():
                if value is not None:
                    metrics_str += f"- {key.upper()}: {value}\n"
        
        # Build conversation history string
        history_str = ""
        if conversation_history:
            history_str = "\n\nPrevious Conversation:\n"
            for msg in conversation_history[-3:]:  # Last 3 messages
                history_str += f"{msg['role']}: {msg['content']}\n"
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst assistant specializing in private equity fund performance.

Your role:
- Answer questions about fund performance using provided context
- Calculate metrics like DPI, IRR when asked
- Explain complex financial terms in simple language
- Always cite your sources from the provided documents

When calculating:
- Use the provided metrics data
- Show your work step-by-step
- Explain any assumptions made

Format your responses:
- Be concise but thorough
- Use bullet points for lists
- Bold important numbers using **number**
- Provide context for metrics"""),
            ("user", """Context from documents:
{context}
{metrics}
{history}

Question: {query}

Please provide a helpful answer based on the context and metrics provided.""")
        ])
        
        # Generate response
        messages = prompt.format_messages(
            context=context_str,
            metrics=metrics_str,
            history=history_str,
            query=query
        )
        
        try:
            logger.debug("Invoking LLM...")
            response = self.llm.invoke(messages)
            if hasattr(response, 'content'):
                logger.info(f"✅ LLM response generated (length: {len(response.content)} chars)")
                return response.content
            logger.info(f"✅ LLM response generated (length: {len(str(response))} chars)")
            return str(response)
        except Exception as e:
            logger.error(f"❌ Error generating LLM response: {str(e)}")
            return f"I apologize, but I encountered an error generating a response: {str(e)}"
