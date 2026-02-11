## pdf2bibtex

An autonomous Python-based pipeline designed to automate the extraction of academic metadata from PDFs and generate accurate BibTeX entries. This project bridges the gap between raw document data and structured research citations using a modular, machine-learning-driven approach.

### Overview 

Researchers often spend significant time on manual metadata extraction. pdf2bibtex simulates a researcher's document analysis process to build a full-stack, autonomous workflow. It transitions from traditional rule-based logic to a machine learning classifier to identify paper titles and metadata with high precision.

### Technical Features 

- Modular Data Pipeline: A multi-stage architecture including sampling, downloading, and loading modules.
- Feature Extraction: Utilizes PyMuPDF to extract visual and textual features from thousands of academic papers.
- ML-Driven Classification: Employs a Random Forest classifier trained on 250,000+ lines of data to identify titles and fields.
- Scalable Architecture: Currently being extended with neural networks and sequence-based models to handle complex fields like author lists and citations.

### Repository Structure

The project is organized into modular scripts to ensure reliability and ease of maintenance:
- src/parser.py: Selects random samples from large paper datasets (e.g., ArXiv).
- src/pdf_downloader.py: Handles the automated downloading of PDF files.
- src/enrich_data.py: Manages the generation of the BibTeX column and data enrichment.
- src/data_builder.py: Constructs the final structured dataset.
- src/core.py: Contains global configurations, file paths, and data schemas.
- src/pdf_loader.py: Manages the loading and initial processing of PDF content.

### Installation & Usage 

Clone the repository:

    git clone https://github.com/madeinuzza/pdf2bibtex.git
    cd pdf2bibtex

Install dependencies: (Recommended: Use a virtual environment)

    pip install -r requirements.txt

Run the pipeline: The core logic is triggered through the modular scripts in src/. Ensure your environment variables and paths are set in core.py.

### Current Status 

- Completed: Automated PDF sampling and downloading, initial metadata extraction using PyMuPDF, and Random Forest title classification.

- In Progress: Implementation of neural networks for sequence labeling to improve author and citation extraction.