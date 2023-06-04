import config, utils
from tqdm.auto import tqdm
import pandas as pd
import ijson
import os

def get_raw_labels():
  with tqdm(unit=' items', position=0, leave=True, ncols=config.tqdm_ncols,
    desc='Loading {}'.format(config.raw_labels_filepath)) as pbar:
    labels = pd.read_csv(
      config.raw_labels_filepath,
      skiprows=lambda x: pbar.update(1) and False
    )
  labels.dropna(inplace=True)
  labels.rename(columns={
      'category_1': 'category',
      'category_2': 'subcategory',
  }, inplace=True)
  labels = labels.reset_index(drop=True)
  return labels


def get_raw_captions_stream():
  captions_chunk = []
  with open(config.raw_captions_filepath, 'r') as raw_captions_reader:
    stream = ijson.kvitems(raw_captions_reader, '')

    with tqdm(unit=' items', position=0, leave=True,
      desc='Loading {}'.format(config.raw_captions_filepath)
    ) as pbar:
      for video_id, caption in stream:
        caption_row = {
          'text': caption,
          'video_id': video_id
        }

        captions_chunk.append(caption_row)
        if len(captions_chunk) >= config.categorizer_chunksize:
          yield captions_chunk
          pbar.update(config.categorizer_chunksize)
          captions_chunk = []

      if len(captions_chunk) > 0:
        yield captions_chunk
        pbar.update(len(captions_chunk))
  

def merge_labels_and_captions_stream(labels, captions_stream):
  for captions_chunk in captions_stream:
    captions_chunk_dataframe = pd.DataFrame(captions_chunk)
    raw_dataset_chunk = pd.merge(
      labels, captions_chunk_dataframe, how='inner', on=['video_id']
    )
    yield raw_dataset_chunk


def get_categorized_dataset_stream(dataset_stream):
  for dataset_chunk in dataset_stream:
    categorized_dataset_chunk = dataset_chunk.groupby('category')
    yield categorized_dataset_chunk


def prepare_raw_categorized_data_files(categories, columns):
  empty_df = pd.DataFrame({}, columns=columns)
  for category in categories:
    filename = utils.get_filename(category, ext='csv')
    filepath = os.path.join(config.raw_data_directory, filename)

    empty_df.to_csv(filepath, mode='w', index=False)


def save_categorized_dataset_stream(categorized_raw_dataset_stream):
  for dataset_chunk in categorized_raw_dataset_stream:
    for category, category_dataset in dataset_chunk:
      filename = utils.get_filename(category, ext='csv')
      filepath = os.path.join(config.raw_data_directory, filename)

      category_dataset.to_csv(filepath, mode='a', header=False, index=False)


def categorize_data():
  print('Categorizing raw data...')

  labels = get_raw_labels()
  columns = pd.Index(labels.columns.to_list() + ['text'])
  categories = list(labels['category'].unique())
  prepare_raw_categorized_data_files(categories, columns)
  
  captions_stream = get_raw_captions_stream()
  merged_dataset_stream = merge_labels_and_captions_stream(labels, captions_stream)
  categorized_dataset_stream = get_categorized_dataset_stream(merged_dataset_stream)
  save_categorized_dataset_stream(categorized_dataset_stream)

  print('Finished categorized raw data!')


if __name__ == '__main__':
  categorize_data()