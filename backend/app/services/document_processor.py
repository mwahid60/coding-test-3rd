"""
Document processing service using pdfplumber

This module handles:
- PDF parsing and table extraction
- Table classification and data extraction
- Text chunking for vector storage
- Error handling and validation
"""
from typing import Dict, List, Any, Optional
import pdfplumber
import re
from sqlalchemy.orm import Session
from app.core.config import settings
from app.services.table_parser import TableParser
from app.services.vector_store import VectorStore
from app.db.session import SessionLocal
from app.models.transaction import CapitalCall, Distribution, Adjustment


class DocumentProcessor:
    """Process PDF documents and extract structured data"""
    
    def __init__(self):
        self.table_parser = TableParser()
        self.vector_store = VectorStore()
        self.chunk_size = 1000  # characters
        self.chunk_overlap = 200  # characters
    
    async def process_document(self, file_path: str, document_id: int, fund_id: int) -> Dict[str, Any]:
        """
        Process a PDF document
        
        Steps:
        1. Open PDF with pdfplumber
        2. Extract tables from each page
        3. Classify and parse tables
        4. Store structured data in database
        5. Extract text and create chunks
        6. Return processing statistics
        
        Args:
            file_path: Path to the PDF file
            document_id: Database document ID
            fund_id: Fund ID
            
        Returns:
            Processing result with statistics
        """
        stats = {
            "status": "completed",
            "pages_processed": 0,
            "tables_found": 0,
            "capital_calls": 0,
            "distributions": 0,
            "adjustments": 0,
            "text_chunks": 0,
            "errors": []
        }
        
        db = SessionLocal()
        
        try:
            # Open PDF
            with pdfplumber.open(file_path) as pdf:
                stats["pages_processed"] = len(pdf.pages)
                
                all_text_content = []
                all_tables = []
                
                # Process each page
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract tables
                        tables = page.extract_tables()
                        
                        if tables:
                            for table in tables:
                                if table and len(table) > 1:
                                    all_tables.append({
                                        "table": table,
                                        "page": page_num + 1,
                                        "context": page.extract_text()[:500]  # First 500 chars for context
                                    })
                                    stats["tables_found"] += 1
                        
                        # Extract text (excluding tables area)
                        page_text = page.extract_text()
                        if page_text:
                            all_text_content.append({
                                "text": page_text,
                                "page": page_num + 1,
                                "document_id": document_id
                            })
                    
                    except Exception as e:
                        stats["errors"].append(f"Page {page_num + 1}: {str(e)}")
                        continue
                
                # Process tables and classify them
                for table_info in all_tables:
                    try:
                        table_type = self.table_parser.classify_table(
                            table_info["table"], 
                            table_info["context"]
                        )
                        
                        if table_type == "capital_call":
                            records = self.table_parser.parse_capital_call_table(
                                table_info["table"], 
                                fund_id
                            )
                            
                            # Save to database
                            for record in records:
                                capital_call = CapitalCall(**record)
                                db.add(capital_call)
                            
                            stats["capital_calls"] += len(records)
                        
                        elif table_type == "distribution":
                            records = self.table_parser.parse_distribution_table(
                                table_info["table"], 
                                fund_id
                            )
                            
                            # Save to database
                            for record in records:
                                distribution = Distribution(**record)
                                db.add(distribution)
                            
                            stats["distributions"] += len(records)
                        
                        elif table_type == "adjustment":
                            records = self.table_parser.parse_adjustment_table(
                                table_info["table"], 
                                fund_id
                            )
                            
                            # Save to database
                            for record in records:
                                adjustment = Adjustment(**record)
                                db.add(adjustment)
                            
                            stats["adjustments"] += len(records)
                    
                    except Exception as e:
                        stats["errors"].append(f"Table processing: {str(e)}")
                        continue
                
                # Commit database changes
                db.commit()
                
                # Chunk text for vector storage
                text_chunks = self._chunk_text(all_text_content)
                stats["text_chunks"] = len(text_chunks)
                
                # Store chunks in vector database (with error handling)
                embedding_errors = 0
                for chunk in text_chunks:
                    try:
                        await self.vector_store.add_document(
                            content=chunk["text"],
                            metadata={
                                "document_id": chunk["document_id"],
                                "fund_id": fund_id,
                                "page": chunk["page"],
                                "char_count": chunk["char_count"]
                            }
                        )
                    except Exception as e:
                        embedding_errors += 1
                        # Don't add all errors to avoid spam, just count them
                        if embedding_errors <= 3:
                            stats["errors"].append(f"Vector store error: {str(e)}")
                
                # Log summary if there were many errors
                if embedding_errors > 3:
                    stats["errors"].append(f"Total vector store errors: {embedding_errors}. Embeddings may not be available for search.")
            
            if stats["errors"]:
                stats["status"] = "completed_with_errors"
        
        except Exception as e:
            stats["status"] = "failed"
            stats["error"] = str(e)
        
        finally:
            db.close()
        
        return stats
    
    def _chunk_text(self, text_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk text content for vector storage
        
        Strategy:
        - Split text into chunks of ~1000 characters
        - Overlap chunks by 200 characters for context
        - Preserve sentence boundaries
        - Add metadata (page number, document ID)
        
        Args:
            text_content: List of text content with metadata
            
        Returns:
            List of text chunks with metadata
        """
        chunks = []
        
        for content in text_content:
            text = content["text"]
            page = content["page"]
            document_id = content["document_id"]
            
            # Clean text
            text = self._clean_text(text)
            
            if not text:
                continue
            
            # Split into sentences
            sentences = self._split_into_sentences(text)
            
            current_chunk = ""
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # If adding this sentence exceeds chunk size
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append({
                        "text": current_chunk.strip(),
                        "page": page,
                        "document_id": document_id,
                        "char_count": len(current_chunk)
                    })
                    
                    # Start new chunk with overlap
                    # Take last N characters for context
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                    current_length = len(current_chunk)
                else:
                    # Add sentence to current chunk
                    current_chunk += " " + sentence if current_chunk else sentence
                    current_length += sentence_length
            
            # Add remaining chunk
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "page": page,
                    "document_id": document_id,
                    "char_count": len(current_chunk)
                })
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,;:!?()\[\]{}"\'-]', '', text)
        
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with nltk or spacy)
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_endings.split(text)
        
        return [s.strip() for s in sentences if s.strip()]
