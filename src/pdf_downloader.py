
import sys
import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from pdf2bibtex.core import TRAIN_DATA_PATH, RAW_PDF_DIR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))



def download_papers():
    # Setup local storage
    if not os.path.exists(RAW_PDF_DIR):
        os.makedirs(RAW_PDF_DIR)
    
    # Load the data you (just enriched)
    df = pd.read_json(TRAIN_DATA_PATH, lines=True)
    print(f"Total papers to verify/download: {len(df)}")

    
    download_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading PDFs"):
        # Force the ID to be a string and strip any weird whitespace
        paper_id = str(row['id']).strip()

        # Skip invalid IDs
        if paper_id == "nan" or not paper_id:
            continue
            
        # Construct the local file path
        safe_id = paper_id.replace('/', '_')
        file_path = os.path.join(RAW_PDF_DIR, f"{safe_id}.pdf")

        # Skip if we already have it
        if os.path.exists(file_path):
            continue

        # ArXiv PDF URL format
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        
        try:
            headers = {'User-Agent': 'EducationalProject/1.0 (contact: your@email.com)'}
            response = requests.get(pdf_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                download_count += 1
                # 3 seconds is the "polite" minimum.
                time.sleep(3) 
            elif response.status_code == 403:
                print("\n[!] Access Forbidden. You are likely rate-limited. Stopping.")
                break
            else:
                print(f"\n[!] Failed {paper_id} (Status: {response.status_code})")

        except Exception as e:
            print(f"\n[!] Error with {paper_id}: {e}")
            time.sleep(5)

    print(f"\nFinished! Downloaded {download_count} new papers.")

if __name__ == "__main__":
    download_papers()