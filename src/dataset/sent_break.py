import config, utils
from tqdm.auto import tqdm
import pandas as pd
import nltk
import os


def break_row_by_sentence(row: pd.Series):
  row['text'] = nltk.sent_tokenize(row['text'])
  return row


def break_dataset_by_sentence(dataset: pd.DataFrame):
  return dataset.progress_apply(break_row_by_sentence, axis=1)


def break_data_by_sentence():
  nltk.download('punkt')
  utils.prepare_directory(config.sentencebroken_data_directory)

  datasets = utils.get_datasets(
    config.clean_data_directory,
    desc='Loading datasets',
    verbose=True
  )
  for category, dataset in tqdm(
    datasets.items(),
    desc='Breaking datasets by sentence',
    ncols=config.tqdm_ncols,
    position=0,
    leave=True
  ):
    sentencebroken_filename = utils.get_filename(category, ext='csv')
    sentencebroken_filepath = os.path.join(
      config.sentencebroken_data_directory,
      sentencebroken_filename
    )
    if os.path.exists(sentencebroken_filepath) and not config.sentbreak_overwrite:
      continue

    tqdm.pandas(desc=category, position=1, ncols=config.tqdm_ncols, leave=False)
    sentencebroken_dataset = break_dataset_by_sentence(dataset)
    sentencebroken_dataset.to_csv(
      sentencebroken_filepath,
      mode='w', index=False
    )

  print('Finished breaking datasets by sentence!')


if __name__ == '__main__':
  break_data_by_sentence()