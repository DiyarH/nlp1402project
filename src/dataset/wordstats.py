import config
from tqdm.auto import tqdm
import pandas as pd


class WordStatistics():
  unique_words = {}
  all_unique_words = {}
  common_words = {}
  noncommon_words = {}
  categorywise_frequencies = {}

  def __init__(self, datasets):
    self.extract_unique_words(datasets)
    self.extract_categorywise_frequencies()
    self.extract_common_and_noncommon_words()
  

  def extract_unique_words(self, datasets):
    self.unique_words = {}
    self.all_unique_words = {}
    for category, dataset in tqdm(
      datasets.items(),
      ncols=config.tqdm_ncols,
      desc='Extracting unique words',
      position=0,
      leave=True
    ):
      unique_words = {}
      for sample in tqdm(dataset['text'],
        ncols=config.tqdm_ncols,
        desc=category, position=1, leave=False
      ):
        for word in sample:
          word = word.lower()
          unique_words[word] = unique_words.get(word, 0) + 1
      self.unique_words[category] = unique_words
      for word, count in unique_words.items():
        self.all_unique_words[word] = self.all_unique_words.get(word, 0) + count


  def extract_categorywise_frequencies(self):
    self.categorywise_frequencies = {}
    for unique_words in self.unique_words.values():
      for word in unique_words:
        self.categorywise_frequencies[word] =\
         self.categorywise_frequencies.get(word, 0) + 1


  def extract_common_and_noncommon_words(self):
    self.noncommon_words = {}
    self.common_words = {
      word: count for word, count in self.all_unique_words.items()
      if self.categorywise_frequencies[word] > 1
    }
    all_noncommon_words = set(
      word for word in self.all_unique_words
      if self.categorywise_frequencies[word] == 1
    )
    for category, unique_words in self.unique_words.items():
      noncommon_words = all_noncommon_words.intersection(unique_words.keys())
      self.noncommon_words[category] = {word: self.all_unique_words[word]
                                        for word in noncommon_words}