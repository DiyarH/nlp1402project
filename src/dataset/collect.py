import config, utils
from zipfile import ZipFile
from tqdm.auto import tqdm
import requests
import os


def collect_raw_file(url, filepath, redownload):
  if os.path.isfile(filepath):
    if not redownload:
      print('{} already exists, skipping...'.format(filepath))
      return
    os.remove(filepath)
  print('Downloading {} to {}...'.format(url, filepath))

  response = requests.get(url, stream=True)
  total_size = int(response.headers.get('content-length', 0))
  with tqdm(total=total_size,
    ncols=config.tqdm_ncols,
    unit='iB',
    unit_scale=True,
    position=0,
    leave=True
  ) as pbar:
    with open(filepath, 'wb') as file:
      for data in response.iter_content(config.collector_chunksize):
        size = file.write(data)
        pbar.update(size)
  

def extract_data_csv_from_zip():
  if not os.path.isfile(config.raw_labels_zip_filepath):
    print('ERROR: {} doesn\'t exist'.format(config.raw_labels_zip_filepath))
    return
  
  if os.path.isfile(config.raw_labels_filepath):
    if not config.collector_reextract_labels:
      print('{} already exists, skipping...'.format(config.raw_labels_filepath))
      return
    os.remove(config.raw_labels_filepath)

  print('Extracting {} from {} to {}...'.format(
    config.raw_labels_url,
    config.raw_labels_zip_filepath,
    config.raw_labels_filepath
  ))
  with ZipFile(config.raw_labels_zip_filepath) as zip:
    with zip.open(config.raw_labels_url) as f_in:
      with open(config.raw_labels_filepath, 'wb') as f_out:
        f_out.write(f_in.read())
  
  if config.collector_delete_labels_zip:
    print('Deleting {}...'.format(config.raw_labels_zip_filepath))
    os.remove(config.raw_labels_zip_filepath)


def collect_data():
  print('Collecting the raw data...')
  utils.prepare_directory(config.raw_data_directory)

  collect_raw_file(config.raw_captions_url, config.raw_captions_filepath, config.collector_redownload_captions)
  collect_raw_file(config.raw_labels_zip_url, config.raw_labels_zip_filepath, config.collector_redownload_labels)
  extract_data_csv_from_zip()

  print('Finished collecting the raw data!')


if __name__ == '__main__':
  collect_data()