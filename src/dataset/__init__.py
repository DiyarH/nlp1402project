import sys
import os
filepath = os.path.abspath(__file__)
filepath = os.path.dirname(filepath)
filepath = os.path.dirname(filepath)
sys.path.append(filepath)

from dataset.config import clean_data_directory
from dataset.utils import get_datasets
from datasets import load_dataset, Dataset
import pandas as pd

def load_dataset_from_huggingface():
  return load_dataset('diyarhamedi/HowTo100M-subtitles-small')

def load_dataset_from_files():
  datasets = get_datasets(clean_data_directory)
  merged_dataset = pd.concat(datasets.values(), ignore_index=True)
  return Dataset.from_pandas(merged_dataset)

def save_dataset_to_huggingface(dataset, token):
  dataset.push_to_hub('diyarhamedi/HowTo100M-subtitles-small', token=token)
