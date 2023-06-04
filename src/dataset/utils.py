import config
import os
from tqdm.auto import tqdm
import pandas as pd
from ast import literal_eval


def prepare_directory(directory):
  print('Preparing {} directory...'.format(directory))
  os.makedirs(directory, exist_ok=True)


def format_category_name(category):
  category = category.replace('&', 'and')
  return category


def get_filename(category, ext=''):
  category = format_category_name(category)
  category = category.lower()
  category = category.replace(' ', '_')
  filename = '.'.join((category, ext))
  return filename


def is_csv_file(filepath):
  return os.path.isfile(filepath) and os.path.splitext(filepath)[1] == '.csv'


def get_csv_filepaths(directory):
  filepaths = list(os.path.join(directory, filename) for filename in os.listdir(directory))
  csv_filepaths = list(filepath for filepath in filepaths if is_csv_file(filepath))
  return csv_filepaths


def get_datasets(directory, verbose=False, broken=False, desc=''):
  datasets = {}

  filepaths = get_csv_filepaths(directory)
  for filepath in tqdm(filepaths,
    desc=desc,
    ncols=config.tqdm_ncols,
    disable=not verbose,
    position=0,
  leave=True):
    converters = {'text': literal_eval} if broken else {}
    dataset = pd.read_csv(
      filepath,
      converters=converters
    )
    category = dataset['category'][0]
    category = format_category_name(category)
    datasets[category] = dataset
  
  return datasets