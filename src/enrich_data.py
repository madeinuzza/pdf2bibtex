import pandas as pd
import json
from pdf2bibtex.core import ArxivPaper
from pdf2bibtex.core import TRAIN_DATA_PATH

def enrich_gold_standard():
    print(f"Loading data from {TRAIN_DATA_PATH}...")
    
    # Read the existing sampled data
    df = pd.read_json(TRAIN_DATA_PATH, lines=True)
    
    # Generate BibTeX for each row using our Data Schema
    def get_bib(row):
        paper = ArxivPaper.from_dict(row)
        return paper.generate_bibtex_entry()
    
    df['bibtex'] = df.apply(get_bib, axis=1)
    
    # Save back to the same file (locking the gold standard)
    df.to_json(TRAIN_DATA_PATH, orient='records', lines=True)
    print("Enrichment complete! BibTeX column added.")

if __name__ == "__main__":
    enrich_gold_standard()



#### SANITY CHECK ####
import pandas as pd
df = pd.read_json("data/processed/arXiv_v1_03-02-2026.jsonl", lines=True)

print("Columns found:", df.columns)
if 'bibtex' in df.columns:
    print("Sample BibTeX:")
    print(df['bibtex'].iloc[0])
else:
    print("Column NOT found. Something went wrong during saving.")