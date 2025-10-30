"""
Query Engine with Intent Classification and Routing

This module handles:
- Intent classification (calculation, definition, retrieval, explanation)
- Query routing to appropriate services (SQL, RAG, Metrics)
- Response formatting and aggregation
- Source citation
"""
from typing import Dict, Any, Optional
from enum import Enum
import re
from sqlalchemy.orm import Session
from app.services.rag_engine import RAGEngine
from app.services.metrics_calculator import MetricsCalculator
from app.db.session import SessionLocal
from app.models.transaction import CapitalCall, Distribution, Adjustment


class QueryIntent(str, Enum):
    """Query intent types"""
    CALCULATION = "calculation"  # Calculate metrics (DPI, IRR, etc.)
    DEFINITION = "definition"    # Define terms (What is DPI?)
    RETRIEVAL = "retrieval"      # Retrieve data (Show capital calls)
    EXPLANATION = "explanation"  # Explain concepts or results


class QueryEngine:
    """
    Main query engine that orchestrates all query processing
    
    Flow:
    1. Classify query intent
    2. Route to appropriate service
    3. Format response
    4. Add citations
    """
    
    def __init__(self, db: Session = None):
        self.db = db or SessionLocal()
        self.rag_engine = RAGEngine()
        self.metrics_calculator = MetricsCalculator(self.db)
        
        # Keywords for intent classification
        self.calculation_keywords = [
            "calculate", "compute", "what is the", "current", "total",
            "dpi", "irr", "pic", "tvpi", "rvpi", "performance"
        ]
        
        self.definition_keywords = [
            "what is", "what does", "define", "meaning of", "explain",
            "definition", "means", "stand for"
        ]
        
        self.retrieval_keywords = [
            "show", "list", "display", "get", "find", "retrieve",
            "all", "capital calls", "distributions", "adjustments"
        ]
    
    async def process_query(
        self, 
        query: str, 
        fund_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a user query
        
        Args:
            query: User's question
            fund_id: Optional fund ID filter
            
        Returns:
            Structured response with answer, intent, and sources
        """
        try:
            # Classify intent
            intent = self._classify_intent(query)
            
            # Route based on intent
            if intent == QueryIntent.CALCULATION:
                result = await self._handle_calculation(query, fund_id)
            
            elif intent == QueryIntent.DEFINITION:
                result = await self._handle_definition(query, fund_id)
            
            elif intent == QueryIntent.RETRIEVAL:
                result = await self._handle_retrieval(query, fund_id)
            
            else:  # EXPLANATION
                result = await self._handle_explanation(query, fund_id)
            
            # Add metadata
            result["intent"] = intent
            result["query"] = query
            result["fund_id"] = fund_id
            
            return result
        
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "intent": "error",
                "query": query,
                "fund_id": fund_id,
                "sources": [],
                "success": False
            }
    
    def _classify_intent(self, query: str) -> QueryIntent:
        """
        Classify query intent using keyword matching
        
        Args:
            query: User's question
            
        Returns:
            Query intent
        """
        query_lower = query.lower()
        
        # Check for metric calculation requests
        metric_patterns = [
            r"what (?:is|was) the (?:current )?(?:dpi|irr|pic|tvpi)",
            r"calculate (?:the )?(?:dpi|irr|pic|tvpi)",
            r"show me the (?:dpi|irr|pic|tvpi)",
            r"current (?:dpi|irr|pic|tvpi)"
        ]
        
        for pattern in metric_patterns:
            if re.search(pattern, query_lower):
                return QueryIntent.CALCULATION
        
        # Check for definition requests
        if any(keyword in query_lower for keyword in ["what is", "what does", "define", "meaning"]):
            # If asking about a metric name (not value)
            if not any(word in query_lower for word in ["the dpi", "the irr", "current", "calculate"]):
                return QueryIntent.DEFINITION
        
        # Check for data retrieval
        if any(keyword in query_lower for keyword in self.retrieval_keywords):
            return QueryIntent.RETRIEVAL
        
        # Default to explanation
        return QueryIntent.EXPLANATION
    
    async def _handle_calculation(self, query: str, fund_id: Optional[int]) -> Dict[str, Any]:
        """Handle calculation queries"""
        
        if not fund_id:
            return {
                "answer": "Please specify a fund ID to calculate metrics.",
                "success": False,
                "sources": []
            }
        
        query_lower = query.lower()
        
        try:
            # Determine which metric to calculate
            if "dpi" in query_lower:
                metrics = self.metrics_calculator.calculate_dpi(fund_id)
                metric_name = "DPI"
            
            elif "irr" in query_lower:
                metrics = self.metrics_calculator.calculate_irr(fund_id)
                metric_name = "IRR"
            
            elif "pic" in query_lower:
                metrics = self.metrics_calculator.calculate_pic(fund_id)
                metric_name = "PIC"
            
            else:
                # Calculate all metrics
                dpi_result = self.metrics_calculator.calculate_dpi(fund_id)
                irr_result = self.metrics_calculator.calculate_irr(fund_id)
                
                answer = f"""**Fund Performance Metrics:**

**DPI (Distribution to Paid-In Capital):** {dpi_result['dpi']:.4f}
- Total Distributions: ${dpi_result['total_distributions']:,.2f}
- Paid-In Capital: ${dpi_result['pic']:,.2f}

**IRR (Internal Rate of Return):** {irr_result['irr']:.2%}

These metrics indicate the fund's performance in returning capital to investors."""
                
                return {
                    "answer": answer,
                    "calculation": {
                        "dpi": dpi_result,
                        "irr": irr_result
                    },
                    "success": True,
                    "sources": []
                }
            
            # Format single metric response
            answer = self._format_metric_answer(metric_name, metrics)
            
            # Get explanation from RAG
            explanation = await self.rag_engine.explain_calculation(
                metric_name, 
                fund_id, 
                metrics
            )
            
            return {
                "answer": answer,
                "explanation": explanation.get("answer", ""),
                "calculation": metrics,
                "success": True,
                "sources": explanation.get("sources", [])
            }
        
        except Exception as e:
            return {
                "answer": f"Error calculating metrics: {str(e)}",
                "success": False,
                "sources": []
            }
    
    async def _handle_definition(self, query: str, fund_id: Optional[int]) -> Dict[str, Any]:
        """Handle definition queries"""
        
        # Extract term to define
        term = self._extract_term(query)
        
        # Get definition from RAG
        result = await self.rag_engine.answer_definition_query(term, fund_id)
        
        result["success"] = True
        return result
    
    async def _handle_retrieval(self, query: str, fund_id: Optional[int]) -> Dict[str, Any]:
        """Handle data retrieval queries"""
        
        if not fund_id:
            return {
                "answer": "Please specify a fund ID to retrieve data.",
                "success": False,
                "sources": []
            }
        
        query_lower = query.lower()
        
        try:
            # Determine what data to retrieve
            if "capital call" in query_lower:
                data = self._get_capital_calls(fund_id, query)
                data_type = "Capital Calls"
            
            elif "distribution" in query_lower:
                data = self._get_distributions(fund_id, query)
                data_type = "Distributions"
            
            elif "adjustment" in query_lower:
                data = self._get_adjustments(fund_id, query)
                data_type = "Adjustments"
            
            else:
                # Use RAG for complex retrieval
                result = await self.rag_engine.answer_data_query(query, fund_id)
                result["success"] = True
                return result
            
            # Format data response
            answer = self._format_data_response(data_type, data)
            
            return {
                "answer": answer,
                "data": data,
                "success": True,
                "sources": []
            }
        
        except Exception as e:
            return {
                "answer": f"Error retrieving data: {str(e)}",
                "success": False,
                "sources": []
            }
    
    async def _handle_explanation(self, query: str, fund_id: Optional[int]) -> Dict[str, Any]:
        """Handle explanation queries using RAG"""
        
        result = await self.rag_engine.query(query, fund_id, top_k=5)
        result["success"] = True
        return result
    
    def _extract_term(self, query: str) -> str:
        """Extract term to define from query"""
        query_lower = query.lower()
        
        # Common patterns
        patterns = [
            r"what is (?:a |an |the )?(\w+)",
            r"what does (\w+) mean",
            r"define (\w+)",
            r"meaning of (\w+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1).upper()
        
        # Fallback: extract first meaningful word
        words = query_lower.split()
        for word in words:
            if len(word) > 2 and word not in ["what", "is", "the", "does", "mean"]:
                return word.upper()
        
        return query
    
    def _get_capital_calls(self, fund_id: int, query: str) -> list:
        """Retrieve capital calls with optional filtering"""
        
        q = self.db.query(CapitalCall).filter(CapitalCall.fund_id == fund_id)
        
        # Filter by year if mentioned
        year_match = re.search(r"20\d{2}", query)
        if year_match:
            year = int(year_match.group(0))
            q = q.filter(
                CapitalCall.call_date >= f"{year}-01-01",
                CapitalCall.call_date <= f"{year}-12-31"
            )
        
        results = q.order_by(CapitalCall.call_date).all()
        
        return [
            {
                "date": str(result.call_date),
                "type": result.call_type,
                "amount": float(result.amount),
                "description": result.description
            }
            for result in results
        ]
    
    def _get_distributions(self, fund_id: int, query: str) -> list:
        """Retrieve distributions with optional filtering"""
        
        q = self.db.query(Distribution).filter(Distribution.fund_id == fund_id)
        
        # Filter by year if mentioned
        year_match = re.search(r"20\d{2}", query)
        if year_match:
            year = int(year_match.group(0))
            q = q.filter(
                Distribution.distribution_date >= f"{year}-01-01",
                Distribution.distribution_date <= f"{year}-12-31"
            )
        
        # Filter by recallable if mentioned
        if "recallable" in query.lower():
            q = q.filter(Distribution.is_recallable == True)
        
        results = q.order_by(Distribution.distribution_date).all()
        
        return [
            {
                "date": str(result.distribution_date),
                "type": result.distribution_type,
                "amount": float(result.amount),
                "recallable": result.is_recallable,
                "description": result.description
            }
            for result in results
        ]
    
    def _get_adjustments(self, fund_id: int, query: str) -> list:
        """Retrieve adjustments with optional filtering"""
        
        q = self.db.query(Adjustment).filter(Adjustment.fund_id == fund_id)
        
        results = q.order_by(Adjustment.adjustment_date).all()
        
        return [
            {
                "date": str(result.adjustment_date),
                "type": result.adjustment_type,
                "amount": float(result.amount),
                "category": result.category,
                "description": result.description
            }
            for result in results
        ]
    
    def _format_metric_answer(self, metric_name: str, calculation: Dict[str, Any]) -> str:
        """Format metric calculation answer"""
        
        if metric_name == "DPI":
            return f"""**DPI (Distribution to Paid-In Capital): {calculation['dpi']:.4f}**

Calculation:
- Total Distributions: ${calculation['total_distributions']:,.2f}
- Paid-In Capital: ${calculation['pic']:,.2f}
- DPI = Distributions / PIC = {calculation['dpi']:.4f}

This metric shows that ${calculation['dpi']:.2f} has been distributed for every $1.00 of paid-in capital."""
        
        elif metric_name == "IRR":
            return f"""**IRR (Internal Rate of Return): {calculation['irr']:.2%}**

The IRR represents the annualized rate of return for this fund, considering the timing of all cash flows."""
        
        elif metric_name == "PIC":
            return f"""**Paid-In Capital (PIC): ${calculation['pic']:,.2f}**

Calculation:
- Total Capital Calls: ${calculation.get('total_calls', 0):,.2f}
- Adjustments: ${calculation.get('adjustments', 0):,.2f}
- Net PIC: ${calculation['pic']:,.2f}"""
        
        return str(calculation)
    
    def _format_data_response(self, data_type: str, data: list) -> str:
        """Format data retrieval response"""
        
        if not data:
            return f"No {data_type.lower()} found for this fund."
        
        answer = f"**{data_type}** ({len(data)} entries):\n\n"
        
        for i, item in enumerate(data, 1):
            answer += f"{i}. **{item['date']}** - {item.get('type', 'N/A')}\n"
            answer += f"   Amount: ${item['amount']:,.2f}\n"
            if item.get('description'):
                answer += f"   {item['description']}\n"
            answer += "\n"
        
        return answer
    
    def get_query_suggestions(self, fund_id: Optional[int] = None) -> list:
        """Get suggested queries for users"""
        
        suggestions = [
            # Calculation queries
            "What is the current DPI of this fund?",
            "Calculate the IRR for this fund",
            "Show me all performance metrics",
            
            # Definition queries
            "What is DPI?",
            "What does IRR mean?",
            "Explain Paid-In Capital",
            
            # Retrieval queries
            "Show me all capital calls in 2024",
            "List all distributions",
            "What were the recallable distributions?",
            
            # Explanation queries
            "How is the fund performing?",
            "Explain the fund's cash flow",
            "What is the investment strategy?"
        ]
        
        return suggestions
