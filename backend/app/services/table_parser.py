"""
Table parser service for extracting and classifying tables from PDFs

This module handles:
- Table classification (capital calls, distributions, adjustments)
- Data extraction and normalization
- Validation and cleaning
"""
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
from decimal import Decimal


class TableParser:
    """Parse and classify tables from PDF documents"""
    
    # Keywords for table classification
    CAPITAL_CALL_KEYWORDS = [
        "capital call", "call", "contribution", "commitment", "capital contributed"
    ]
    
    DISTRIBUTION_KEYWORDS = [
        "distribution", "return", "dividend", "payout", "recallable"
    ]
    
    ADJUSTMENT_KEYWORDS = [
        "adjustment", "rebalance", "clawback", "refund", "correction"
    ]
    
    def __init__(self):
        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # 2024-01-15
            r'\d{2}/\d{2}/\d{4}',  # 01/15/2024
            r'\d{2}-\d{2}-\d{4}',  # 01-15-2024
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # 1/15/24
        ]
    
    def classify_table(self, table: List[List[str]], context: str = "") -> Optional[str]:
        """
        Classify table type based on headers and context
        
        Args:
            table: 2D list representing table data
            context: Surrounding text for context
            
        Returns:
            Table type: 'capital_call', 'distribution', 'adjustment', or None
        """
        if not table or len(table) < 2:
            return None
        
        # Get first row (headers) and convert to lowercase
        headers = [str(cell).lower() for cell in table[0] if cell]
        all_text = " ".join(headers) + " " + context.lower()
        
        # Check for capital call indicators
        capital_call_score = sum(
            1 for keyword in self.CAPITAL_CALL_KEYWORDS 
            if keyword in all_text
        )
        
        # Check for distribution indicators
        distribution_score = sum(
            1 for keyword in self.DISTRIBUTION_KEYWORDS 
            if keyword in all_text
        )
        
        # Check for adjustment indicators
        adjustment_score = sum(
            1 for keyword in self.ADJUSTMENT_KEYWORDS 
            if keyword in all_text
        )
        
        # Return classification based on highest score
        scores = {
            'capital_call': capital_call_score,
            'distribution': distribution_score,
            'adjustment': adjustment_score
        }
        
        max_score = max(scores.values())
        if max_score == 0:
            return None
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def parse_capital_call_table(self, table: List[List[str]], fund_id: int) -> List[Dict[str, Any]]:
        """
        Parse capital call table
        
        Expected columns: Date, Call Number/Type, Amount, Description
        """
        if not table or len(table) < 2:
            return []
        
        headers = [str(cell).lower() for cell in table[0]]
        data_rows = table[1:]
        
        # Find column indices
        date_idx = self._find_column_index(headers, ["date", "call date"])
        amount_idx = self._find_column_index(headers, ["amount", "call amount"])
        type_idx = self._find_column_index(headers, ["type", "call type", "call number", "number"])
        desc_idx = self._find_column_index(headers, ["description", "desc", "details", "notes"])
        
        capital_calls = []
        
        for row in data_rows:
            if not row or len(row) < 2:
                continue
            
            try:
                # Extract date
                call_date = self._parse_date(row[date_idx] if date_idx is not None else row[0])
                if not call_date:
                    continue
                
                # Extract amount
                amount = self._parse_amount(row[amount_idx] if amount_idx is not None else row[1])
                if amount is None or amount <= 0:
                    continue
                
                # Extract type
                call_type = str(row[type_idx]) if type_idx is not None and type_idx < len(row) else "Capital Call"
                
                # Extract description
                description = str(row[desc_idx]) if desc_idx is not None and desc_idx < len(row) else ""
                
                capital_calls.append({
                    "fund_id": fund_id,
                    "call_date": call_date,
                    "call_type": call_type.strip(),
                    "amount": float(amount),
                    "description": description.strip()
                })
                
            except Exception as e:
                print(f"Error parsing capital call row: {e}")
                continue
        
        return capital_calls
    
    def parse_distribution_table(self, table: List[List[str]], fund_id: int) -> List[Dict[str, Any]]:
        """
        Parse distribution table
        
        Expected columns: Date, Type, Amount, Recallable, Description
        """
        if not table or len(table) < 2:
            return []
        
        headers = [str(cell).lower() for cell in table[0]]
        data_rows = table[1:]
        
        # Find column indices
        date_idx = self._find_column_index(headers, ["date", "distribution date"])
        amount_idx = self._find_column_index(headers, ["amount", "distribution amount"])
        type_idx = self._find_column_index(headers, ["type", "distribution type"])
        recallable_idx = self._find_column_index(headers, ["recallable", "recall"])
        desc_idx = self._find_column_index(headers, ["description", "desc", "details", "notes"])
        
        distributions = []
        
        for row in data_rows:
            if not row or len(row) < 2:
                continue
            
            try:
                # Extract date
                dist_date = self._parse_date(row[date_idx] if date_idx is not None else row[0])
                if not dist_date:
                    continue
                
                # Extract amount
                amount = self._parse_amount(row[amount_idx] if amount_idx is not None else row[1])
                if amount is None or amount <= 0:
                    continue
                
                # Extract type
                dist_type = str(row[type_idx]) if type_idx is not None and type_idx < len(row) else "Return"
                
                # Extract recallable status
                is_recallable = False
                if recallable_idx is not None and recallable_idx < len(row):
                    recallable_text = str(row[recallable_idx]).lower()
                    is_recallable = recallable_text in ["yes", "true", "1", "y"]
                
                # Extract description
                description = str(row[desc_idx]) if desc_idx is not None and desc_idx < len(row) else ""
                
                distributions.append({
                    "fund_id": fund_id,
                    "distribution_date": dist_date,
                    "distribution_type": dist_type.strip(),
                    "is_recallable": is_recallable,
                    "amount": float(amount),
                    "description": description.strip()
                })
                
            except Exception as e:
                print(f"Error parsing distribution row: {e}")
                continue
        
        return distributions
    
    def parse_adjustment_table(self, table: List[List[str]], fund_id: int) -> List[Dict[str, Any]]:
        """
        Parse adjustment table
        
        Expected columns: Date, Type, Category, Amount, Description
        """
        if not table or len(table) < 2:
            return []
        
        headers = [str(cell).lower() for cell in table[0]]
        data_rows = table[1:]
        
        # Find column indices
        date_idx = self._find_column_index(headers, ["date", "adjustment date"])
        amount_idx = self._find_column_index(headers, ["amount", "adjustment amount"])
        type_idx = self._find_column_index(headers, ["type", "adjustment type"])
        category_idx = self._find_column_index(headers, ["category", "class"])
        desc_idx = self._find_column_index(headers, ["description", "desc", "details", "notes"])
        
        adjustments = []
        
        for row in data_rows:
            if not row or len(row) < 2:
                continue
            
            try:
                # Extract date
                adj_date = self._parse_date(row[date_idx] if date_idx is not None else row[0])
                if not adj_date:
                    continue
                
                # Extract amount (can be negative)
                amount = self._parse_amount(row[amount_idx] if amount_idx is not None else row[1], allow_negative=True)
                if amount is None:
                    continue
                
                # Extract type
                adj_type = str(row[type_idx]) if type_idx is not None and type_idx < len(row) else "Adjustment"
                
                # Extract category
                category = str(row[category_idx]) if category_idx is not None and category_idx < len(row) else ""
                
                # Determine if it's a contribution adjustment
                is_contribution_adjustment = "capital call" in adj_type.lower() or "contribution" in adj_type.lower()
                
                # Extract description
                description = str(row[desc_idx]) if desc_idx is not None and desc_idx < len(row) else ""
                
                adjustments.append({
                    "fund_id": fund_id,
                    "adjustment_date": adj_date,
                    "adjustment_type": adj_type.strip(),
                    "category": category.strip(),
                    "amount": float(amount),
                    "is_contribution_adjustment": is_contribution_adjustment,
                    "description": description.strip()
                })
                
            except Exception as e:
                print(f"Error parsing adjustment row: {e}")
                continue
        
        return adjustments
    
    def _find_column_index(self, headers: List[str], keywords: List[str]) -> Optional[int]:
        """Find column index by matching keywords"""
        for i, header in enumerate(headers):
            header_lower = header.lower()
            if any(keyword in header_lower for keyword in keywords):
                return i
        return None
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse date string to ISO format (YYYY-MM-DD)"""
        if not date_str:
            return None
        
        date_str = str(date_str).strip()
        
        # Try to match date patterns
        for pattern in self.date_patterns:
            match = re.search(pattern, date_str)
            if match:
                date_part = match.group(0)
                
                try:
                    # Try different date formats
                    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y"]:
                        try:
                            parsed_date = datetime.strptime(date_part, fmt)
                            return parsed_date.strftime("%Y-%m-%d")
                        except ValueError:
                            continue
                except Exception:
                    pass
        
        return None
    
    def _parse_amount(self, amount_str: str, allow_negative: bool = False) -> Optional[Decimal]:
        """Parse amount string to decimal"""
        if not amount_str:
            return None
        
        # Convert to string and clean
        amount_str = str(amount_str).strip()
        
        # Remove currency symbols and commas
        amount_str = re.sub(r'[$,€£¥]', '', amount_str)
        
        # Handle parentheses as negative
        is_negative = False
        if amount_str.startswith('(') and amount_str.endswith(')'):
            is_negative = True
            amount_str = amount_str[1:-1]
        
        # Extract numeric value
        match = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', amount_str)
        if not match:
            return None
        
        try:
            amount = Decimal(match.group(0).replace(',', ''))
            
            if is_negative:
                amount = -amount
            
            if not allow_negative and amount < 0:
                return None
            
            return amount
            
        except Exception:
            return None
