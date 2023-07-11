from wordstats import WordStatistics
import config, utils
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from math import log
import pandas as pd
import os


class Statistics:
    wordstats = None
    raw_dataset_size = {}
    clean_dataset_size = {}
    n_sentences = {}
    n_words = {}
    n_unique_words = {}
    word_frequencies = {}
    word_frequencies_in_category = {}
    word_RNFs = {}
    word_TF_IDFs = {}

    def save_dataset_sizes_stats(self):
        stats_filename = "dataset_sizes.csv"
        stats_filepath = os.path.join(config.stats_directory, stats_filename)
        raw_dataset_size_stats = pd.DataFrame(
            self.raw_dataset_size.items(), columns=["category", "raw dataset size"]
        )
        clean_dataset_size_stats = pd.DataFrame(
            self.clean_dataset_size.items(), columns=["category", "clean dataset size"]
        )
        dataset_sizes_stats = raw_dataset_size_stats.merge(
            clean_dataset_size_stats, how="inner", on=["category"]
        )
        dataset_sizes_stats.to_csv(stats_filepath, mode="w", index=False)

    def save_n_sentences_stats(self):
        stats_filename = "number_of_sentences.csv"
        stats_filepath = os.path.join(config.stats_directory, stats_filename)
        n_sentences_stats = pd.DataFrame(
            self.n_sentences.items(), columns=["category", "number of sentences"]
        )
        n_samples = pd.DataFrame(
            self.clean_dataset_size.items(), columns=["category", "number of samples"]
        )
        avg_n_sentences = (
            n_sentences_stats["number of sentences"]
            / n_samples["number of samples"].values
        )
        avg_n_sentences = avg_n_sentences.apply(lambda x: round(x, 2))
        n_sentences_stats["avg. number of sentences"] = avg_n_sentences
        n_sentences_stats.to_csv(stats_filepath, mode="w", index=False)

    def save_n_words_stats(self):
        stats_filename = "number_of_words.csv"
        stats_filepath = os.path.join(config.stats_directory, stats_filename)
        n_words_stats = pd.DataFrame(
            self.n_words.items(), columns=["category", "number of words"]
        )
        n_samples = pd.DataFrame(
            self.clean_dataset_size.items(), columns=["category", "number of samples"]
        )
        avg_n_words = (
            n_words_stats["number of words"] / n_samples["number of samples"].values
        )
        avg_n_words = avg_n_words.apply(lambda x: round(x, 2))
        n_words_stats["avg. number of words"] = avg_n_words
        n_words_stats.to_csv(stats_filepath, mode="w", index=False)

    def save_n_unique_words_stats(self):
        stats_filename = "number_of_unique_words.csv"
        stats_filepath = os.path.join(config.stats_directory, stats_filename)
        n_unique_words_stats = pd.DataFrame(
            self.n_unique_words.items(), columns=["category", "number of unique words"]
        )
        n_unique_words_stats.to_csv(stats_filepath, mode="w", index=False)

    def save_most_frequent_words_in_category(self, top_n=10):
        most_frequent_words_directory = os.path.join(
            config.stats_directory, "most_frequent_words"
        )
        utils.prepare_directory(most_frequent_words_directory)
        for category, word_frequencies in tqdm(
            self.wordstats.noncommon_words.items(),
            ncols=config.tqdm_ncols,
            desc="Saving most frequent words",
            position=0,
            leave=True,
        ):
            most_frequent_words = sorted(
                word_frequencies.items(), key=lambda item: item[1], reverse=True
            )[:top_n]
            words, frequencies = zip(*most_frequent_words)
            stats_filename = utils.get_filename(category, ext="png")
            stats_filepath = os.path.join(most_frequent_words_directory, stats_filename)
            plt.figure(figsize=(5, 10))
            plt.title(category)
            plt.bar(words, frequencies)
            plt.xticks(rotation=90, fontsize=10)
            plt.savefig(stats_filepath)
            plt.close()

    def save_RNF_top_words(self, top_n=10):
        rnf_top_words_directory = os.path.join(config.stats_directory, "RNF_top_words")
        utils.prepare_directory(rnf_top_words_directory)
        for category, word_rnfs in tqdm(
            self.word_RNFs.items(),
            ncols=config.tqdm_ncols,
            desc="Saving words with highest RNF",
            position=0,
            leave=True,
        ):
            top_words = sorted(
                word_rnfs.items(), key=lambda item: item[1], reverse=True
            )[:top_n]
            words, rnfs = zip(*top_words)
            stats_filename = utils.get_filename(category, ext="png")
            stats_filepath = os.path.join(rnf_top_words_directory, stats_filename)
            plt.figure(figsize=(5, 10))
            plt.title(category)
            plt.bar(words, rnfs)
            plt.xticks(rotation=90, fontsize=10)
            plt.savefig(stats_filepath)
            plt.close()

    def save_TF_IDF_top_words(self, top_n=10):
        tf_idf_top_words_directory = os.path.join(
            config.stats_directory, "TF_IDF_top_words"
        )
        utils.prepare_directory(tf_idf_top_words_directory)
        for category, word_tf_idfs in tqdm(
            self.word_TF_IDFs.items(),
            ncols=config.tqdm_ncols,
            desc="Saving words with highest TF-IDF",
            position=0,
            leave=True,
        ):
            top_words = sorted(
                word_tf_idfs.items(), key=lambda item: item[1], reverse=True
            )[:top_n]
            words, tf_idfs = zip(*top_words)
            stats_filename = utils.get_filename(category, ext="png")
            stats_filepath = os.path.join(tf_idf_top_words_directory, stats_filename)
            plt.figure(figsize=(5, 10))
            plt.title(category)
            plt.bar(words, tf_idfs)
            plt.xticks(rotation=90, fontsize=10)
            plt.savefig(stats_filepath)
            plt.close()

    def save_word_frequencies_histogram(self, top_n=100):
        most_frequent_words = sorted(
            self.wordstats.all_unique_words.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]
        words, frequencies = zip(*most_frequent_words)
        stats_filename = "frequency_histogram.png"
        stats_filepath = os.path.join(config.stats_directory, stats_filename)
        plt.figure(figsize=(20, 7))
        plt.bar(words, frequencies)
        plt.xticks(rotation=90, fontsize=10)
        plt.savefig(stats_filepath)
        plt.close()

    def save_all(self):
        utils.prepare_directory(config.stats_directory)
        print("Saving the statistics...")

        if len(self.raw_dataset_size) > 0 and len(self.clean_dataset_size) > 0:
            self.save_dataset_sizes_stats()
        if len(self.n_sentences) > 0:
            self.save_n_sentences_stats()
        if len(self.n_words) > 0:
            self.save_n_words_stats()
        if len(self.n_unique_words) > 0:
            self.save_n_unique_words_stats()
        if self.wordstats is not None:
            self.save_word_frequencies_histogram(config.n_top_words_histogram)
            self.save_most_frequent_words_in_category(config.n_top_words_frequency)
        if len(self.word_RNFs) > 0:
            self.save_RNF_top_words(config.n_top_words_RNF)
        if len(self.word_TF_IDFs) > 0:
            self.save_TF_IDF_top_words(config.n_top_words_TF_IDF)

        print("Finished saving the statistics!")


class StatisticsExtractor:
    clean_datasets = None
    sentencebroken_datasets = None
    wordbroken_datasets = None

    def load_clean_datasets(self):
        self.clean_datasets = utils.get_datasets(
            config.clean_data_directory, desc="Loading clean datasets", verbose=True
        )

    def load_sentencebroken_datasets(self):
        self.sentencebroken_datasets = utils.get_datasets(
            config.sentencebroken_data_directory,
            desc="Loading sentencebroken datasets",
            verbose=True,
            broken=True,
        )

    def load_wordbroken_datasets(self):
        self.wordbroken_datasets = utils.get_datasets(
            config.wordbroken_data_directory,
            desc="Loading wordbroken datasets",
            verbose=True,
            broken=True,
        )

    def get_dataset_stream_info(self, dataset_stream: pd.DataFrame):
        dataset_size = 0
        with tqdm(ncols=config.tqdm_ncols, position=1, leave=False) as pbar:
            for chunk in dataset_stream:
                dataset_size += len(chunk.index)
                category = chunk["category"].iloc[0]
                category = utils.format_category_name(category)
                pbar.update(len(chunk.index))
                pbar.set_description(category)
        return category, dataset_size

    def get_dataset_size(self, dataset: pd.DataFrame):
        return len(dataset.index)

    def get_n_pieces(self, category, broken_dataset: pd.DataFrame):
        tqdm.pandas(ncols=config.tqdm_ncols, desc=category, position=1, leave=False)
        return broken_dataset.progress_apply(lambda s: len(s["text"]), axis=1).sum()

    def extract_raw_dataset_size_per_category(self, stats_obj: Statistics):
        stats = {}
        total_size = 0
        filepaths = utils.get_csv_filepaths(config.raw_data_directory)
        if config.raw_labels_filepath in filepaths:
            filepaths.remove(config.raw_labels_filepath)
        for filepath in tqdm(
            filepaths,
            ncols=config.tqdm_ncols,
            desc="Extracting raw dataset size",
            position=0,
            leave=True,
        ):
            dataset_stream = pd.read_csv(
                filepath,
                chunksize=config.stats_chunksize,
            )
            category, size = self.get_dataset_stream_info(dataset_stream)
            stats[category] = size
            total_size += size
        stats["Total"] = total_size
        stats_obj.raw_dataset_size = stats

    def extract_clean_dataset_size_per_category(self, stats_obj: Statistics):
        stats = {}
        total_size = 0
        for category, dataset in tqdm(
            self.clean_datasets.items(),
            ncols=config.tqdm_ncols,
            desc="Extracting clean dataset size",
            position=0,
            leave=True,
        ):
            size = self.get_dataset_size(dataset)
            stats[category] = size
            total_size += size
        stats["Total"] = total_size
        stats_obj.clean_dataset_size = stats

    def extract_n_sentences_per_category(self, stats_obj: Statistics):
        stats = {}
        total_n_sentences = 0

        for category, dataset in tqdm(
            self.sentencebroken_datasets.items(),
            ncols=config.tqdm_ncols,
            desc="Extracting number of sentences",
            position=0,
            leave=True,
        ):
            n_sentences = self.get_n_pieces(category, dataset)
            stats[category] = n_sentences
            total_n_sentences += n_sentences
        stats["Total"] = total_n_sentences
        stats_obj.n_sentences = stats

    def extract_n_words_per_category(self, stats_obj: Statistics):
        stats = {}
        total_n_words = 0
        for category, dataset in tqdm(
            self.wordbroken_datasets.items(),
            ncols=config.tqdm_ncols,
            desc="Extracting number of words",
            position=0,
            leave=True,
        ):
            n_words = self.get_n_pieces(category, dataset)
            stats[category] = n_words
            total_n_words += n_words
        stats["Total"] = total_n_words
        stats_obj.n_words = stats

    def extract_n_unique_words(self, stats_obj: Statistics):
        stats = {}
        stats["All"] = len(stats_obj.wordstats.all_unique_words)
        stats["Common"] = len(stats_obj.wordstats.common_words)
        for category, noncommon_words in stats_obj.wordstats.noncommon_words.items():
            stats[category] = len(noncommon_words)
        stats_obj.n_unique_words = stats

    def calculate_RNF(self, word, category, stats: Statistics):
        return (
            stats.wordstats.unique_words[category][word]
            / stats.wordstats.all_unique_words[word]
        )

    def extract_RNFs(self, stats_obj: Statistics):
        for category, words in (
            pbar := tqdm(
                stats_obj.wordstats.unique_words.items(),
                ncols=config.tqdm_ncols,
                desc="Extracting RNF metric",
                position=0,
                leave=True,
            )
        ):
            common_words = set(stats_obj.wordstats.common_words).intersection(words)
            word_RNFs = tqdm(
                (
                    (word, self.calculate_RNF(word, category, stats_obj))
                    for word in common_words
                ),
                ncols=config.tqdm_ncols,
                total=len(common_words),
                desc=category,
                position=1,
                leave=False,
            )
            stats_obj.word_RNFs[category] = {word: rnf for word, rnf in word_RNFs}

    def calculate_TF_IDF(self, word, category, stats: Statistics):
        tf = stats.wordstats.unique_words[category][word] / stats.n_words[category]
        idf = log(
            len(stats.wordstats.unique_words)
            / stats.wordstats.categorywise_frequencies[word]
        )
        return tf * idf

    def extract_TF_IDFs(self, stats_obj: Statistics):
        for category, words in (
            pbar := tqdm(
                stats_obj.wordstats.unique_words.items(),
                ncols=config.tqdm_ncols,
                desc="Extracting TF-IDF metric",
                position=0,
                leave=True,
            )
        ):
            pbar.set_postfix(category=category)
            common_words = set(stats_obj.wordstats.common_words).intersection(words)
            word_TF_IDFs = tqdm(
                (
                    (word, self.calculate_TF_IDF(word, category, stats_obj))
                    for word in common_words
                ),
                ncols=config.tqdm_ncols,
                total=len(common_words),
                desc=category,
                position=1,
                leave=False,
            )
            stats_obj.word_TF_IDFs[category] = {
                word: tf_idf for word, tf_idf in word_TF_IDFs
            }

    def extract_all(self, stats: Statistics):
        print("Extrating the statistics...")

        self.extract_raw_dataset_size_per_category(stats)
        self.load_clean_datasets()
        self.extract_clean_dataset_size_per_category(stats)

        self.load_sentencebroken_datasets()
        self.extract_n_sentences_per_category(stats)

        self.load_wordbroken_datasets()
        self.extract_n_words_per_category(stats)

        stats.wordstats = WordStatistics(self.wordbroken_datasets)
        self.extract_n_unique_words(stats)

        self.extract_RNFs(stats)
        self.extract_TF_IDFs(stats)

        print("Finished extrating the statistics!")


if __name__ == "__main__":
    stats = Statistics()
    stats_extractor = StatisticsExtractor()
    stats_extractor.extract_all(stats)
    stats.save_all()
