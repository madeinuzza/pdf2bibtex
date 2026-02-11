import json
import random
import pandas as pd
from pdf2bibtex.core import set_seed

# Setting the seed here ensures reproducibility
set_seed(42)

# ------------------------- # Select Random Subset of ArXiv Papers Post-2007 # ------------------------- #

def get_random_post_2007_subset(target_categories, samples_per_cat=500):
    """
    This module selects random papers from different sections in arXiv (the original dataset is huge).
    Random samples are used to make sure the training is not biased. 
    It uses the Reservoir Sampling algorithm (https://en.wikipedia.org/wiki/Reservoir_sampling)
    
    Conditions: 
        - We only choose papers from after 2007, as in April 2007 arXiv updated the id of papers to contain the year
          (for example, ID: 0704.0001 → The first two digits (07) mean 2007.)
        - We only choose papers that have a non-empty journal-ref field (indicating publication)

    :param target_categories: sections from arXiv
    :param samples_per_cat: number of samples per section
    """

    buckets = {cat: [] for cat in target_categories}
    counts_seen = {cat: 0 for cat in target_categories}
    
    # arXiv Dataset: obtained from https://www.kaggle.com/datasets/Cornell-University/arxiv/data
    file_path = 'data/raw/arxiv-metadata-oai-snapshot.json' 
    
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            
            
            # Check if journal-ref is present and not empty
            journal = item.get('journal-ref')
            if not journal or str(journal).strip() == "":
                continue

            # ArXiv ID format post-2007: YYMM.number (e.g., 0901.0123)
            # Old format (pre-2007): category/YYMMNNN (e.g., math/0301001)
            # We filter for IDs that are purely numeric before the dot
            # and ensure the year (first two digits) is >= 07
            paper_id = item['id']
            parts = paper_id.split('.')
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            
            year_val = int(parts[0][:2])
            # Note: IDs starting with 00-06 are from 2000-2006. 
            # 07 and above are 2007+.
            if not (7 <= year_val <= 26):
                continue

            paper_cats = item['categories'].split()
            for cat in paper_cats:
                for target_cat in target_categories:
                    if cat.startswith(target_cat):
                        counts_seen[target_cat] += 1
                    
                        # Reservoir Sampling to keep memory low and selection random
                        if len(buckets[target_cat]) < samples_per_cat:
                            buckets[target_cat].append({
                                'id': paper_id,
                                'authors': item['authors'],
                                'journal-ref': item['journal-ref'],
                                'year': f"20{parts[0][:2]}",
                                'title': item['title'],
                                'abstract': item['abstract'],
                                'section': cat
                            })
                        else:
                            s = random.randint(0, counts_seen[target_cat] - 1)
                            if s < samples_per_cat:
                                buckets[target_cat][s] = {
                                    'id': paper_id,
                                    'authors': item['authors'],
                                    'journal-ref': item['journal-ref'],
                                    'year': f"20{parts[0][:2]}",
                                    'title': item['title'],
                                    'abstract': item['abstract'],
                                    'section': cat
                                }
                        break 

    all_data = [paper for cat_list in buckets.values() for paper in cat_list]
    return pd.DataFrame(all_data)




if __name__ == "__main__": 
    # Create a random subset of ArXiv papers post-2007 & containing journal references 

    # Define target sections
    my_sections = ['cs', 'physics', 'math', 'q-bio', 'q-fin'] # Major sections of ArXiv

    # Get 1000 random papers per category/section from 2007 onwards
    df = get_random_post_2007_subset(my_sections, samples_per_cat=1000)

    # Save to disk
    filename = 'data/processed/arXiv_v1_06-02-2026.jsonl'
    df.to_json(filename, orient='records', lines=True)
    print(f"Gold Standard dataset locked at: {filename}")

    # Quick check on the distribution
    print(df.groupby(['section', 'year']).size().unstack(fill_value=0))


