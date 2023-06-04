import config, utils
from tqdm.auto import tqdm
import pandas as pd
import unicodedata
import random
import os


def get_raw_dataset_subset(dataset_stream):
  n_rows = 0
  samples = []

  with tqdm(
    desc='Loading',
    ncols=config.tqdm_ncols,
    position=1,
    leave=False
  ) as pbar:
    for dataset_chunk in dataset_stream:
      category = dataset_chunk['category'].iloc[0]
      pbar.set_description(category)
      for i in range(len(dataset_chunk.index)):
        if len(samples) < config.max_n_samples_per_category:
          samples.append(dataset_chunk.iloc[i])
          continue
        rnd = random.randint(0, i + n_rows)
        if rnd < config.max_n_samples_per_category:
          samples[rnd] = dataset_chunk.iloc[i]
      
      n_rows += len(dataset_chunk.index)
      pbar.update(len(dataset_chunk.index))

  dataset_subset = pd.DataFrame(samples)
  return dataset_subset


def cleanup_dataset(category, dataset: pd.DataFrame):
  def cleanup_row(row: pd.Series):
    clean_text = ''
    for char in row['text']:
      if unicodedata.category(char) in ['Zs', 'Nd', 'Lu', 'Ll', 'Po']:
        clean_text += char
    row['text'] = clean_text
    return row
  tqdm.pandas(position=1, desc=category, leave=False)
  return dataset.progress_apply(cleanup_row, axis=1)


def save_clean_dataset(category, dataset):
  filename = utils.get_filename(category, ext='csv')
  clean_dataset_filepath = os.path.join(config.clean_data_directory, filename)
  dataset.to_csv(clean_dataset_filepath, mode='w', index=False)


def cleanup_data(seed=None):
  random.seed(seed)
  utils.prepare_directory(config.clean_data_directory)
  filepaths = utils.get_csv_filepaths(config.raw_data_directory)
  filepaths.remove(config.raw_labels_filepath)

  for filepath in tqdm(
    filepaths, position=0, leave=True,
    ncols=config.tqdm_ncols,
    desc='Cleaning up raw data'
  ):
    dataset_stream = pd.read_csv(
      filepath,
      chunksize=config.cleaner_chunksize,
    )
    dataset_subset = get_raw_dataset_subset(dataset_stream)
    category = dataset_subset['category'].iloc[0]
    clean_dataset = cleanup_dataset(category, dataset_subset)
    save_clean_dataset(category, clean_dataset)

  print('Finished cleaning up the raw data!')


if __name__ == '__main__':
  cleanup_data(config.sampling_seed)