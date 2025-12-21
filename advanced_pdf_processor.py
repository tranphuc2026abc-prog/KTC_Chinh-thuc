"""
Advanced PDF Processor for Vietnamese Educational Textbooks
===========================================================
Author: Senior NLP Engineer
Purpose: National Science & Technology Competition - RAG Chatbot Upgrade

This module replaces naive RecursiveCharacterTextSplitter with:
- Context-aware hierarchical document segmentation
- Precise metadata extraction (chapter, lesson, page)
- Noise reduction (headers, footers, page numbers)
- State-machine based content tracking

Target: Tin 10_KNTT.pdf (Vietnamese Computer Science Grade 10)
"""

import re
import fitz  # PyMuPDF
from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document
import unicodedata


class VietnameseTextbookProcessor:
    """
    Advanced processor for Vietnamese textbook PDFs with hierarchical structure.
    Implements context-aware chunking with proper metadata tracking.
    """
    
    # Noise patterns to filter out
    NOISE_PATTERNS = [
        r'KẾT\s+NỐI\s+TRI\s+THỨC\s+VỚI\s+CUỘC\s+SỐNG',
        r'TIN\s+HỌC\s+\d+',
        r'CHƯƠNG\s+TRÌNH\s+GIÁO\s+DỤC',
        r'PHÂN\s+PHỐI\s+CHƯƠNG\s+TRÌNH',
        r'^\s*\d+\s*$',  # Isolated page numbers
    ]
    
    # Structural patterns for Vietnamese textbooks
    TOPIC_PATTERN = re.compile(
        r'(?:^|\n)\s*CHỦ\s+ĐỀ\s+(\d+)[\.:\s]*(.*?)(?:\n|$)',
        re.IGNORECASE | re.MULTILINE
    )
    
    LESSON_PATTERN = re.compile(
        r'(?:^|\n)\s*BÀI\s+(\d+)[\.:\s]*(.*?)(?:\n|$)',
        re.IGNORECASE | re.MULTILINE
    )
    
    def __init__(self, pdf_path: str):
        """
        Initialize the processor with a PDF file path.
        
        Args:
            pdf_path: Path to the Vietnamese textbook PDF
        """
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.source_name = pdf_path.split('/')[-1]
        
    def _normalize_text(self, text: str) -> str:
        """
        Normalize Vietnamese text (NFC normalization, whitespace cleanup).
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Normalized text string
        """
        # Unicode normalization for Vietnamese
        text = unicodedata.normalize('NFC', text)
        
        # Replace non-breaking spaces and zero-width characters
        text = text.replace('\xa0', ' ').replace('\u200b', '')
        
        # Normalize multiple whitespaces
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def _is_noise(self, text: str) -> bool:
        """
        Check if a text line is noise (header/footer/page number).
        
        Args:
            text: Text line to check
            
        Returns:
            True if text is noise, False otherwise
        """
        text_clean = text.strip()
        
        # Empty or very short lines
        if len(text_clean) < 3:
            return True
        
        # Check against noise patterns
        for pattern in self.NOISE_PATTERNS:
            if re.search(pattern, text_clean, re.IGNORECASE):
                return True
        
        # Isolated numbers (likely page numbers)
        if text_clean.isdigit() and len(text_clean) <= 3:
            return True
        
        return False
    
    def _extract_page_text(self, page_num: int) -> Tuple[str, List[str]]:
        """
        Extract clean text from a PDF page, filtering noise.
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            Tuple of (full_page_text, list_of_lines)
        """
        page = self.doc[page_num]
        text = page.get_text()
        
        # Normalize the text
        text = self._normalize_text(text)
        
        # Split into lines and filter noise
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not self._is_noise(line):
                clean_lines.append(line)
        
        full_text = '\n'.join(clean_lines)
        return full_text, clean_lines
    
    def _detect_topic(self, text: str) -> Optional[str]:
        """
        Detect "Chủ đề" (Topic/Chapter) from text.
        
        Args:
            text: Text to search in
            
        Returns:
            Topic string in format "Chủ đề X. NAME" or None
        """
        match = self.TOPIC_PATTERN.search(text)
        if match:
            topic_num = match.group(1).strip()
            topic_name = match.group(2).strip()
            return f"Chủ đề {topic_num}. {topic_name}"
        return None
    
    def _detect_lesson(self, text: str) -> Optional[str]:
        """
        Detect "Bài" (Lesson) from text.
        
        Args:
            text: Text to search in
            
        Returns:
            Lesson string in format "Bài X. NAME" or None
        """
        match = self.LESSON_PATTERN.search(text)
        if match:
            lesson_num = match.group(1).strip()
            lesson_name = match.group(2).strip()
            return f"Bài {lesson_num}. {lesson_name}"
        return None
    
    def _split_into_semantic_chunks(
        self, 
        text: str, 
        max_chunk_size: int = 1000
    ) -> List[str]:
        """
        Split text into semantic chunks respecting paragraph boundaries.
        
        Args:
            text: Text to split
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_len = len(para)
            
            # If single paragraph exceeds max_chunk_size, force split
            if para_len > max_chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long paragraph by sentences
                sentences = re.split(r'([.!?]+\s+)', para)
                temp_chunk = ""
                for i, sent in enumerate(sentences):
                    if len(temp_chunk) + len(sent) > max_chunk_size and temp_chunk:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = sent
                    else:
                        temp_chunk += sent
                
                if temp_chunk.strip():
                    chunks.append(temp_chunk.strip())
                    
            # Normal paragraph handling
            elif current_length + para_len + 2 > max_chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_len
            else:
                current_chunk.append(para)
                current_length += para_len + 2  # +2 for \n\n
        
        # Add remaining
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def process_pdf_advanced(
        self, 
        chunk_size: int = 1000,
        overlap: int = 100
    ) -> List[Document]:
        """
        Main processing function: Extract PDF with context-aware hierarchical chunking.
        
        This is the CORE function that replaces the naive text splitter.
        
        Algorithm:
        1. Iterate through all PDF pages
        2. Extract and clean text from each page
        3. Maintain state machine for current topic/lesson context
        4. Detect structural changes (new topic, new lesson)
        5. Create chunks with proper metadata enrichment
        
        Args:
            chunk_size: Maximum characters per chunk (default: 1000)
            overlap: Character overlap between chunks (default: 100)
            
        Returns:
            List of LangChain Document objects with enriched metadata
        """
        documents = []
        
        # State machine variables
        current_topic = None
        current_lesson = None
        
        # Buffer for accumulating content within a lesson
        content_buffer = []
        buffer_page_start = 0
        
        print(f"📚 Processing: {self.source_name}")
        print(f"📄 Total pages: {len(self.doc)}")
        
        for page_num in range(len(self.doc)):
            page_text, lines = self._extract_page_text(page_num)
            
            if not page_text.strip():
                continue
            
            # Detect structural changes on this page
            detected_topic = self._detect_topic(page_text)
            detected_lesson = self._detect_lesson(page_text)
            
            # STATE TRANSITION: New Topic detected
            if detected_topic:
                # Commit previous lesson's content
                if content_buffer:
                    self._commit_buffer_to_documents(
                        documents,
                        content_buffer,
                        current_topic,
                        current_lesson,
                        buffer_page_start,
                        page_num - 1,
                        chunk_size,
                        overlap
                    )
                    content_buffer = []
                
                # Update state
                current_topic = detected_topic
                current_lesson = None  # Reset lesson when new topic starts
                buffer_page_start = page_num
                
                print(f"  📌 Page {page_num + 1}: Detected {current_topic}")
            
            # STATE TRANSITION: New Lesson detected
            if detected_lesson:
                # Commit previous lesson's content
                if content_buffer:
                    self._commit_buffer_to_documents(
                        documents,
                        content_buffer,
                        current_topic,
                        current_lesson,
                        buffer_page_start,
                        page_num - 1,
                        chunk_size,
                        overlap
                    )
                    content_buffer = []
                
                # Update state
                current_lesson = detected_lesson
                buffer_page_start = page_num
                
                print(f"    📖 Page {page_num + 1}: Detected {current_lesson}")
            
            # Accumulate content for current context
            content_buffer.append({
                'text': page_text,
                'page': page_num
            })
        
        # Commit remaining buffer
        if content_buffer:
            self._commit_buffer_to_documents(
                documents,
                content_buffer,
                current_topic,
                current_lesson,
                buffer_page_start,
                len(self.doc) - 1,
                chunk_size,
                overlap
            )
        
        print(f"✅ Generated {len(documents)} context-aware chunks")
        return documents
    
    def _commit_buffer_to_documents(
        self,
        documents: List[Document],
        buffer: List[Dict],
        topic: Optional[str],
        lesson: Optional[str],
        page_start: int,
        page_end: int,
        chunk_size: int,
        overlap: int
    ):
        """
        Convert accumulated buffer into Document objects with metadata.
        
        Args:
            documents: List to append new documents to
            buffer: Accumulated content buffer
            topic: Current topic/chapter
            lesson: Current lesson
            page_start: Starting page number
            page_end: Ending page number
            chunk_size: Maximum chunk size
            overlap: Overlap between chunks
        """
        if not buffer:
            return
        
        # Combine all text from buffer
        full_text = '\n\n'.join([item['text'] for item in buffer])
        
        # Get representative page (middle of range)
        representative_page = page_start + (page_end - page_start) // 2
        
        # Split into semantic chunks
        chunks = self._split_into_semantic_chunks(full_text, chunk_size)
        
        # Create overlapping chunks if needed
        final_chunks = []
        for i, chunk in enumerate(chunks):
            # Add overlap from previous chunk
            if i > 0 and overlap > 0:
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
                chunk = overlap_text + '\n' + chunk
            
            final_chunks.append(chunk)
        
        # Create Document objects with proper metadata
        for chunk_idx, chunk_text in enumerate(final_chunks):
            # Build metadata following UI contract
            metadata = {
                'source': self.source_name,
                'page': representative_page + 1,  # 1-indexed for display
                'chapter': topic if topic else 'Nội dung chung',
                'lesson': lesson if lesson else 'Phần giới thiệu',
                'chunk_index': chunk_idx,
                'total_chunks': len(final_chunks),
                'page_range': f"{page_start + 1}-{page_end + 1}"
            }
            
            # Create Document object
            doc = Document(
                page_content=chunk_text.strip(),
                metadata=metadata
            )
            
            documents.append(doc)
    
    def close(self):
        """Close the PDF document."""
        if self.doc:
            self.doc.close()


def process_pdf_advanced(pdf_path: str, chunk_size: int = 1000, overlap: int = 100) -> List[Document]:
    """
    Standalone function to process a Vietnamese textbook PDF with advanced chunking.
    
    This function is the drop-in replacement for the naive processing pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Maximum characters per chunk (default: 1000)
        overlap: Character overlap between chunks (default: 100)
    
    Returns:
        List of LangChain Document objects with enriched metadata
    
    Example:
        >>> documents = process_pdf_advanced("PDF_KNOWLEDGE/Tin 10_KNTT.pdf")
        >>> print(documents[0].metadata)
        {'source': 'Tin 10_KNTT.pdf', 'page': 5, 'chapter': 'Chủ đề 1. ...', 'lesson': 'Bài 1. ...'}
    """
    processor = VietnameseTextbookProcessor(pdf_path)
    try:
        documents = processor.process_pdf_advanced(chunk_size, overlap)
        return documents
    finally:
        processor.close()


# ============================================================================
# DEMONSTRATION & TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 80)
    print("Vietnamese Textbook Advanced PDF Processor - Demo")
    print("=" * 80)
    
    # Check if PDF path is provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Default test path
        pdf_path = "PDF_KNOWLEDGE/Tin 10_KNTT.pdf"
    
    print(f"\n🔍 Testing with: {pdf_path}\n")
    
    try:
        # Process the PDF
        documents = process_pdf_advanced(pdf_path, chunk_size=1000, overlap=100)
        
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"✅ Total documents generated: {len(documents)}")
        
        # Show metadata statistics
        topics = set()
        lessons = set()
        for doc in documents:
            if doc.metadata.get('chapter'):
                topics.add(doc.metadata['chapter'])
            if doc.metadata.get('lesson'):
                lessons.add(doc.metadata['lesson'])
        
        print(f"📚 Unique topics detected: {len(topics)}")
        print(f"📖 Unique lessons detected: {len(lessons)}")
        
        # Show first 3 documents
        print("\n" + "=" * 80)
        print("SAMPLE DOCUMENTS (First 3)")
        print("=" * 80)
        
        for i, doc in enumerate(documents[:3]):
            print(f"\n--- Document {i + 1} ---")
            print(f"Source: {doc.metadata['source']}")
            print(f"Page: {doc.metadata['page']}")
            print(f"Chapter: {doc.metadata['chapter']}")
            print(f"Lesson: {doc.metadata['lesson']}")
            print(f"Content preview (first 200 chars):")
            print(doc.page_content[:200] + "...")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
