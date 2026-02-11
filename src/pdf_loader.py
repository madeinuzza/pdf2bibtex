import fitz  # This is PyMuPDF
from typing import List, Dict, Any, cast
from pdf2bibtex.core import PDFLine 
import json

"""     
    PDFLoader is responsible for loading and parsing PDF documents.
    It uses PyMuPDF (fitz) to read PDF files and extract text along with metadata
    such as font size and style (bold/italic).
"""

class PDFLoader:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def get_first_page_lines(self) -> List[PDFLine]:
        """
        Extracts lines from the first page with their metadata.
        """
        page = self.doc[0]  # Get the first page of the PDF
        page_dict = cast(Dict[str, Any], page.get_text("dict")) # type: ignore
        blocks = page_dict.get("blocks", []) 
        # PyMuPDF's get_text("dict") provides various levels of detail:
        # “blocks”: generate a list of text blocks (= paragraphs). Each block contains lines, and each line contains spans (with font info).
        # “html”: creates a full visual version of the page including any images. This can be displayed with your internet browser.
        # “dict” / “json”: same information level as HTML, but provided as a Python dictionary or resp. JSON string. See TextPage.extractDICT() for details of its structure.
        
        lines_data = []
        # Before the loop
        page_height = page.rect.height


        for b in blocks:
            # Tell Pylance b is a dictionary
            if not isinstance(b, dict) or "lines" not in b:
                continue

            if "lines" not in b: # Some blocks might be images or other non-text elements, so we skip those.
                continue

            for l in b["lines"]:
                # Combine all spans in this line first
                line_text_parts = []
                max_font_size = 0.0
                any_bold = False
                # Use the y-position of the first span as the line's y
                first_span_y = l["spans"][0].get("bbox", (0, 0, 0, 0))[1]

                for s in l["spans"]:
                    text_part = s.get("text", "")
                    line_text_parts.append(text_part)
                    
                    # Track highest font size and if ANY part is bold
                    max_font_size = max(max_font_size, float(s.get("size", 0)))
                    if int(s.get("flags", 0)) & 16:
                        any_bold = True

                full_line_text = " ".join(line_text_parts).strip()
                if not full_line_text:
                    continue

                # Create ONE PDFLine for the entire line
                normalized_y = first_span_y / page_height
                
                line_obj = PDFLine(
                    text=full_line_text,
                    page_number=0,
                    line_index=len(lines_data),
                    y_position=normalized_y,
                    font_size=max_font_size,
                    is_bold=any_bold,
                )
                lines_data.append(line_obj)   
                    
        return lines_data
    

    
    def get_title_candidate(self, lines: List[PDFLine]) -> str:
        # Filter out known noise and restrict to the top of the page
        # We ignore the very top (headers) and anything below 35% of the page
        title_zone_lines = [
            l for l in lines 
            if "arXiv" not in l.text 
            and 0.08 < l.y_position < 0.35  # Adjusted for normalized y
        ]
        
        if not title_zone_lines:
            return ""
        
        # Find the max font size in THIS specific zone
        max_size = max(l.font_size for l in title_zone_lines)
        
        # Only join lines that match that max size within this zone
        title_parts = [
            l.text for l in title_zone_lines 
            if abs(l.font_size - max_size) < 0.1
        ]
        
        return " ".join(title_parts)

    def close(self):
        self.doc.close()



def get_true_title(arxiv_id, metadata_path):
    if not os.path.exists(metadata_path):
        return f"File Not Found: {metadata_path}"
        
    with open(metadata_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Force the ID to a string to avoid the 'float' error
            meta_id = str(entry.get('id', ''))
            
            # Now .startswith() will work safely
            if meta_id.startswith(str(arxiv_id)):
                return " ".join(entry['title'].split()).strip()
    return "ID Not in JSON"
   


if __name__ == "__main__":
    from pdf2bibtex.core import RAW_PDF_DIR, TRAIN_DATA_PATH
    import os
    
    metadata_path = TRAIN_DATA_PATH # os.path.join(BASE_DIR, "data", "processed", "arXiv_v1_06-02-2026.jsonl")
    test_files = [f for f in os.listdir(RAW_PDF_DIR) if f.endswith('.pdf')]
    
    print(f"Checking metadata at: {TRAIN_DATA_PATH}") 
    print(f"Metadata file exists: {os.path.exists(TRAIN_DATA_PATH)}")

    for filename in test_files[:3]:
        arxiv_id = filename.replace('.pdf', '')
        loader = PDFLoader(os.path.join(RAW_PDF_DIR, filename))
        lines = loader.get_first_page_lines()
        
        pred = loader.get_title_candidate(lines)
        true = get_true_title(arxiv_id, metadata_path)
        
        print(f"\nID: {arxiv_id}")
        print(f"PRED: {pred}")
        print(f"TRUE: {true}")
        print("-" * 30)