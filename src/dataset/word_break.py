import config, utils
from tqdm.auto import tqdm
import pandas as pd
import nltk
import os


def break_row_by_word(row: pd.Series):
  wordbroken_text = row['text']
  for char in ['.', ',', '?', '!']:
    wordbroken_text = wordbroken_text.replace(char, '')
  wordbroken_text = nltk.word_tokenize(wordbroken_text)
  row['text'] = wordbroken_text
  return row


def break_dataset_by_word(dataset: pd.DataFrame):
  return dataset.progress_apply(break_row_by_word, axis=1)


def break_data_by_word():
  nltk.download('punkt')
  utils.prepare_directory(config.wordbroken_data_directory)

  datasets = utils.get_datasets(
    config.clean_data_directory,
    desc='Loading datasets',
    verbose=True
  )
  for category, dataset in tqdm(
    datasets.items(),
    desc='Breaking datasets by word',
    ncols=config.tqdm_ncols,
    position=0,
    leave=True
  ):
    wordbroken_filename = utils.get_filename(category, ext='csv')
    wordbroken_filepath = os.path.join(
      config.wordbroken_data_directory,
      wordbroken_filename
    )
    if os.path.exists(wordbroken_filepath) and not config.wordbreak_overwrite:
      continue

    tqdm.pandas(desc=category, position=1, ncols=config.tqdm_ncols, leave=False)
    wordbroken_dataset = break_dataset_by_word(dataset)
    wordbroken_dataset.to_csv(
      wordbroken_filepath,
      mode='w', index=False
    )

  print('Finished breaking datasets by word!')


if __name__ == '__main__':
  break_data_by_word()