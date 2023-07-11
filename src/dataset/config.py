import os
import json

filepath = os.path.abspath(__file__)
src_dataset_directory = os.path.dirname(filepath)
src_directory = os.path.dirname(src_dataset_directory)
root_directory = os.path.dirname(src_directory)
config_filepath = os.path.join(root_directory, 'config.json')

with open(config_filepath, 'r') as file:
  config = json.load(file)

stats_directory = os.path.join(
  root_directory, config['stats_directory']
)

config = config['dataset']
raw_data_directory = os.path.join(
  root_directory, config['paths']['raw_data']
)
clean_data_directory = os.path.join(
  root_directory, config['paths']['clean_data']
)
sentencebroken_data_directory = os.path.join(
  root_directory, config['paths']['sentencebroken_data']
)
wordbroken_data_directory = os.path.join(
  root_directory, config['paths']['wordbroken_data']
)

raw_labels_filepath = os.path.join(raw_data_directory, config['filenames']['raw_labels'])
raw_captions_filepath = os.path.join(raw_data_directory, config['filenames']['raw_captions'])
raw_labels_zip_filepath = os.path.join(raw_data_directory, config['filenames']['raw_labels_zip'])

raw_labels_url = config['urls']['raw_labels']
raw_captions_url = config['urls']['raw_captions']
raw_labels_zip_url = config['urls']['raw_labels_zip']

collector_chunksize = config['chunksize']['collector'] * 1024
categorizer_chunksize = config['chunksize']['categorizer'] * 1024
cleaner_chunksize = config['chunksize']['cleaner'] * 1024
stats_chunksize = config['chunksize']['stats'] * 1024

max_n_samples_per_category = config['max_n_samples_per_category']
sampling_seed = config['sampling_seed']

n_top_words_frequency = config['stats_n_top_words']['frequency']
n_top_words_RNF = config['stats_n_top_words']['RNF']
n_top_words_TF_IDF = config['stats_n_top_words']['TF-IDF']
n_top_words_histogram = config['stats_n_top_words']['frequency_histogram']

collector_redownload_labels = config['collector_behavior']['redownload_labels']
collector_redownload_captions = config['collector_behavior']['redownload_captions']
collector_reextract_labels = config['collector_behavior']['reextract_labels']
collector_delete_labels_zip = config['collector_behavior']['delete_labels_zip']

sentbreak_overwrite = config['sentbreak_overwrite']
wordbreak_overwrite = config['wordbreak_overwrite']

tqdm_ncols = 100