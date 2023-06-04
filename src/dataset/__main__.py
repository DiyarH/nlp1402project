from collect import collect_data
from categorize import categorize_data
from cleanup import cleanup_data
from sent_break import break_data_by_sentence
from word_break import break_data_by_word
from stats import Statistics, StatisticsExtractor


def build_dataset_and_extract_metrics():
  collect_data()
  categorize_data()
  cleanup_data()
  break_data_by_sentence()
  break_data_by_word()

  stats = Statistics()
  stats_extractor = StatisticsExtractor()
  stats_extractor.extract_all(stats)
  stats.save_all()


if __name__ == '__main__':
  build_dataset_and_extract_metrics()