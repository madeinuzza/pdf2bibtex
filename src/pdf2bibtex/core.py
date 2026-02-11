import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd 
import os




# --------------- Configuration Constants --------------- #

# This points to the folder containing core.py (src/pdf2bibtex)
_current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up two levels to reach the PDF2BIBTEX root
BASE_DIR = os.path.abspath(os.path.join(_current_dir, "..", ".."))

# Based on your Screenshot 15.01.58:
RAW_PDF_DIR = os.path.join(BASE_DIR, "data", "raw", "raw_pdfs")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "arXiv_v1_06-02-2026.jsonl")

# --------------- Utility Functions --------------- #

# Sets the random seed for reproducibility across various libraries
def set_seed(seed=42):
    """Sets all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to: {seed}")






# --------------- Data Schemas --------------- #
"""
    In ML projects, passing around raw dictionaries—like data['id']—is risky 
    because if you misspell a key once (e.g., data['Id']), your whole training 
    pipeline crashes.
    To avoid this, we define structured data schemas using dataclasses. This
    way, we get type checking and auto-completion in our IDEs, making our code
    more robust and easier to maintain.
    Each dataclass here represents a specific data structure used in our project.
"""

@dataclass # This is a "blueprint" for your data. By using @dataclass, Python automatically writes standard methods for you (like __init__ and __repr__).
class ArxivPaper:
    """Schema for an arXiv paper entry."""
    id: str
    title: str
    authors: str
    abstract: str
    section: str
    journal_ref: Optional[str]
    year: Optional[int]

    @classmethod
    def from_dict(cls, data: dict):
        # Maps the JSON keys to our class fields
        return cls(
            id=data['id'],
            title=data['title'],
            abstract=data['abstract'],
            section=data['section'],
            year=int(data['year']),
            authors=data['authors'],
            journal_ref=data.get('journal-ref')
        )
    def generate_bibtex_entry(self) -> str:
        """Generates a simple BibTeX entry for the paper."""
        
        # 1. Extract the Last Name of the first author safely
        try:
            # Get the first author's full name chunk
            first_author_raw = self.authors.split(',')[0]
            # Get the last word of that chunk (usually the Last Name)
            last_name = first_author_raw.split()[-1]
            # Clean up LaTeX artifacts and non-alphanumeric chars
            clean_name = "".join(filter(str.isalnum, last_name))
        except (IndexError, AttributeError):
            clean_name = "Unknown"

        # 2. Build the citation key (e.g., Einstein1935)
        cite_key = f"{clean_name}{self.year}"

        # Use journal_ref if available, otherwise fallback to arXiv preprint
        venue = self.journal_ref if self.journal_ref else f"arXiv preprint arXiv:{self.id}"
        
        bib = (
            f"@article{{{cite_key},\n"
            f"  author = {{{self.authors}}},\n"
            f"  title = {{{self.title}}},\n"
            f"  journal = {{{venue}}},\n"
            f"  year = {{{self.year}}},\n"
            f"  note = {{arXiv:{self.id}}}\n"
            f"}}"
        )
        return bib




@dataclass
class PDFLine:
    """A single line extracted from a PDF to be classified by the model."""
    text: str
    page_number: int
    line_index: int       # Position relative to other lines
    y_position: float     # Normalized vertical position (0.0 to 1.0)
    font_size: float
    is_bold: bool
    label: Optional[str] = None  # TITLE, AUTHOR, VENUE, YEAR, or OTHER