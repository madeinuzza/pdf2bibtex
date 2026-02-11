# Copilot Instructions for research-bib-automator

## Project Overview
This is a **neural network-based PDF metadata extraction system** that automatically extracts bibliographic information (title, authors, journal/conference, year) from scientific research PDFs and generates BibTeX entries. The current implementation focuses on arXiv papers but is designed to generalize to journal PDFs.

## Architecture & Data Flow

### Core Pipeline (Sequential)
1. **PDF Ingestion** → PDF text/layout extraction into lines with positional metadata
2. **Candidate Generation** → Convert extracted lines into candidate samples (currently: one label per line)
3. **Feature Extraction** → Convert candidates into numerical vectors combining:
   - **Text features**: bag-of-words, n-grams, character/word length, regex patterns (year-like, "journal", "vol.", "arXiv")
   - **Layout features**: line position, font size, bold/italic flags, center alignment
4. **Neural Classification** → Multi-class classifier (5 classes: TITLE, AUTHORS, VENUE, YEAR, OTHER)
5. **Post-processing** → Author splitting, year parsing, journal abbreviation normalization
6. **BibTeX Generation** → Output structured .bib file entries

### Current Implementation Status
- `parser.py`: Implements **data sampling** from arXiv using Reservoir Sampling algorithm
  - Loads `arxiv-metadata-oai-snapshot.json` (2.4M papers)
  - Filters for post-2007 papers (new ID format: YYMM.number)
  - Stratified random sampling across 5 categories: cs, physics, math, q-bio, q-fin
  - Outputs balanced dataset to `my_arxiv_subset.jsonl`
- **Missing**: PDF parser, feature extraction, model training, inference pipeline, BibTeX formatter

## Key Design Patterns & Conventions

### Multi-Class Classification Approach
- **Problem formulation**: Each line → one of {TITLE, AUTHORS, VENUE, YEAR, OTHER}
- **Baseline model**: Logistic regression on hand-crafted features (sanity check)
- **Production model**: Small MLP (embeddings → dense layers with ReLU → 5-way softmax)
- **Evaluation**: Per-class precision/recall, F1 (especially for TITLE/AUTHORS), confusion matrix

### Feature Engineering Philosophy
- **Template-agnostic design**: Use normalized positions ("top 5% of page") not absolute indices
- **Regex-based text signals**: Year patterns, journal keywords, author name patterns
- **Layout signals**: Relative font size (vs. page median), vertical position, formatting flags
- **Avoid overfitting**: Don't hardcode exact indices or LaTeX-specific assumptions

### Data & Labeling
- **Balanced sampling**: Use stratified random sampling across arXiv categories
- **JSONL format**: Each line is a JSON record `{id, year, title, abstract, section}`
- **Training approach**: Label 30-50 PDFs initially; use mixed arXiv + journal set for fine-tuning

## Critical Workflows

### Running the Data Sampler
```bash
python parser.py
# Reads arxiv-metadata-oai-snapshot.json (2.4M lines)
# Outputs my_arxiv_subset.jsonl with 2500 balanced samples (500 per category)
# Prints distribution table: sections × years
```

### Expected File Dependencies
- `arxiv-metadata-oai-snapshot.json` (must exist; download from Kaggle)
- `dblp-v10.csv` (available but not yet integrated)
- Output: `my_arxiv_subset.jsonl`

## Integration Points & External Dependencies

### arXiv Dataset Specifics
- **ID format change (2007)**: Pre-2007: `category/YYMMNNN`; Post-2007: `YYMM.number`
- **Year encoding**: IDs like `0901.0123` = January 2009; first two digits map to `20XX`
- **Categories**: Multi-category papers tagged with space-separated strings (e.g., "cs.AI cs.LG")
- **Metadata fields**: `id`, `title`, `abstract`, `categories`, `authors` available in source JSON

### Domain Adaptation Strategy (arXiv → Journals)
The architecture anticipates generalization challenges:
- **Potential issues**: Journal PDFs have headers/footers, structured layouts, different typesetting
- **Mitigation**: Use template-agnostic features; add 20-50 labeled journal PDFs for fine-tuning
- **Evaluation plan**: Separate test sets for arXiv-only and journal validation

## Future Implementation Checklist (Phase 2-5)
- [ ] **Phase 2**: Build PDF parser (PyMuPDF); extract lines with layout metadata
- [ ] **Phase 3**: Implement feature extraction module; train logistic regression baseline
- [ ] **Phase 4**: Implement MLP model; add regularization + bias/variance experiments
- [ ] **Phase 5**: Polish CLI, add unit tests, type hints; write training report

## Code Examples & Patterns
- **Reservoir Sampling**: Used in `parser.py` to efficiently handle 2.4M-record arXiv JSON without loading all in memory
- **Stratified sampling**: Ensure representation across 5 categories despite skewed arXiv distribution
- **Year extraction from ID**: `year_val = int(paper_id[:2]); full_year = f"20{year_val:02d}"`
