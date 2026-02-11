import pandas as pd
from pdf2bibtex.core import BASE_DIR
import os



"""
The code below reads the training data we built in data_builder.py and prints out some basic statistics about it.

Printed Output (after running with sample data arXiv_v1_06-02-2026.jsonl):
Total lines extracted: 253925

Label Distribution:
label
OTHER    247397
TITLE      6528
Name: count, dtype: int64

Average Font Size by Label:
label
OTHER    10.111597
TITLE    15.411203
Name: font_size, dtype: float64

This shows that we have 6,528 lines labeled as "TITLE" and 247,397 lines labeled as "OTHER". 
The average font size for "TITLE" lines is significantly larger than for "OTHER" lines, which is a good sign 
that our labeling process is capturing the visual features we intended.
"""

### Simple analysis of the built training data ###
# This script reads the training data JSONL file created in data_builder.py
# and prints out some basic statistics about it.
path = os.path.join(BASE_DIR, "data", "processed", "training_set_v1.jsonl")
df = pd.read_json(path, lines=True)

print(f"Total lines extracted: {len(df)}")
print("\nLabel Distribution:")
print(df['label'].value_counts())

print("\nAverage Font Size by Label:")
print(df.groupby('label')['font_size'].mean())

