"""
PDF extraction service for Arc-Fusion.

This module handles the extraction of text content from PDF files,
separating this responsibility from the main document processor.
"""

import io
from typing import Union
from PyPDF2 import PdfReader
from app.utils.logger import get_logger

logger = get_logger('arc_fusion.document.pdf_extractor')


class PDFExtractor:
    """Service for extracting text content from PDF files."""
    
    def __init__(self, format: str = "text"):
        """
        Initialize the PDF extractor.
        
        Args:
            format: Output format ("text" or "layout")
        """
        self.format = format
    
    def extract_text(self, content: Union[bytes, io.BytesIO]) -> str:
        """
        Extract text content from PDF bytes.
        
        Args:
            content: PDF file content as bytes or BytesIO
            
        Returns:
            Extracted text content
            
        Raises:
            Exception: If PDF extraction fails
        """
        try:
            # Handle both bytes and BytesIO
            if isinstance(content, bytes):
                pdf_stream = io.BytesIO(content)
            else:
                pdf_stream = content
            
            # Open PDF with PyPDF2
            reader = PdfReader(pdf_stream)
            
            # Extract text from all pages
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                text_parts.append(text)
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"Successfully extracted text from PDF with {len(text_parts)} pages")
            
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {str(e)}")
            raise Exception(f"PDF extraction failed: {str(e)}")
    
    def get_page_count(self, content: Union[bytes, io.BytesIO]) -> int:
        """
        Get the number of pages in a PDF.
        
        Args:
            content: PDF file content as bytes or BytesIO
            
        Returns:
            Number of pages in the PDF
        """
        try:
            # Handle both bytes and BytesIO
            if isinstance(content, bytes):
                pdf_stream = io.BytesIO(content)
            else:
                pdf_stream = content
            
            reader = PdfReader(pdf_stream)
            page_count = len(reader.pages)
            
            return page_count
            
        except Exception as e:
            logger.error(f"Failed to get page count from PDF: {str(e)}")
            return 0
