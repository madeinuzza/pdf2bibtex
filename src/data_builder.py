

import os
import json
import dataclasses
from tqdm import tqdm
from pdf2bibtex.core import TRAIN_DATA_PATH, RAW_PDF_DIR, BASE_DIR
from pdf_loader import PDFLoader

def load_title_metadata_to_dict(metadata_path: str) -> dict:
    """
    
    
    :param metadata_path: arXiv_v1_06-02-2026.jsonl file path
    :type metadata_path: str
    :return: 
    :rtype: dict[Any, Any]
    
    Loads metadata once into a dictionary for O(1) lookup speed."""
    titles_map = {}
    with open(metadata_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            
            m_id = str(entry.get('id', ''))
            titles_map[m_id] = " ".join(entry['title'].split()).strip()
    return titles_map

def is_line_in_title(line_text: str, true_title: str) -> bool:
    """Uses 'squashed' text matching to handle spacing issues."""
    line_squashed = "".join(line_text.lower().split())
    true_squashed = "".join(true_title.lower().split())
    if len(line_text.strip()) < 4:
        return False
    return line_squashed in true_squashed

def build_training_data():
    # Setup paths and load the Ground Truth map
    output_path = os.path.join(BASE_DIR, "data", "processed", "training_set_v1.jsonl")
    titles_map = load_title_metadata_to_dict(TRAIN_DATA_PATH)
    pdf_files = [f for f in os.listdir(RAW_PDF_DIR) if f.endswith('.pdf')]
    
    print(f"Starting build for {len(pdf_files)} PDFs...")
    
    with open(output_path, 'w') as f_out:
        # Use tqdm to see a progress bar 
        for filename in tqdm(pdf_files):
            arxiv_id = filename.replace('.pdf', '')
            true_title = titles_map.get(arxiv_id)
            
            if not true_title:
                continue
                
            # Extract visual features using PDFLoader
            try:
                loader = PDFLoader(os.path.join(RAW_PDF_DIR, filename))
                lines = loader.get_first_page_lines()
                loader.close()
                
                # Label each line and save
                for line_obj in lines:
                    # Assign the label based on fuzzy matcher
                    if is_line_in_title(line_obj.text, true_title):
                        line_obj.label = "TITLE"
                    else:
                        line_obj.label = "OTHER"
                    
                    # Convert Dataclass to dict and write one JSON line
                    row = dataclasses.asdict(line_obj)
                    f_out.write(json.dumps(row) + "\n")
                    
            except Exception as e:
                print(f"Error processing {arxiv_id}: {e}")

    print(f"Build complete! Saved to {output_path}")

if __name__ == "__main__":
    build_training_data()