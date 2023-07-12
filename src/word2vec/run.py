import os, sys
filepath = os.path.abspath(__file__)
filepath = os.path.dirname(filepath)
filepath = os.path.dirname(filepath)
sys.path.append(filepath)

from word2vec import *
from sgd import *
import time
import numpy as np
import nltk
from dataset import load_dataset_from_huggingface
from dataset.utils import format_category_name
from config import *

nltk.download('punkt')


def word2vec(dataset, config):
	vec_size = config['vector_size']
	window_size = config['window_size']
	learning_rate = config['learning_rate']
	n_iterations = config['n_iterations']

	sentences = [nltk.word_tokenize(doc) for doc in dataset]
	vocabulary = set([word for sentence in sentences for word in sentence])
	word2ind = {word: index for index, word in enumerate(vocabulary)}
	ind2word = {index: word for index, word in enumerate(vocabulary)}
	vocab_size = len(vocabulary)
	
	startTime=time.time()
	word_vectors = np.concatenate(
		((np.random.rand(vocab_size, vec_size) - 0.5) /
		vec_size, np.zeros((vocab_size, vec_size))),
		axis=0)
	word_vectors = sgd(
		lambda vec: word2vec_sgd_wrapper(skipgram, word2ind, vec, sentences, window_size,
			negSamplingLossAndGradient),
		word_vectors, learning_rate, n_iterations, None, True, PRINT_EVERY=10)

	print('training took %d seconds' % (time.time() - startTime))
	return word_vectors, word2ind


def run():
    if not os.path.exists(models_directory):
        os.mkdir(models_directory)
        
    dataset = load_dataset_from_huggingface()
    train_dataset = dataset['train']
    categories = list(set(train_dataset['category']))
    corpus = [sample['text'] for sample in train_dataset]
    categorized_corpus = {category:
                    	 [sample['text'] for sample in train_dataset
                         if sample['category'] == category]
                         for category in categories}
    for category, categorized in categorized_corpus.items():
        category_name = format_category_name(category)
        filename = '{}.word2vec.npy'.format(category_name)
        filepath = os.path.join(models_directory, filename)
        word_vectors, ind2word = word2vec(categorized, word2vec_config)
        np.save(filepath, word_vectors)
        print('saved word vectors as {}!'.format(filepath))
    
    filename = 'all.word2vec.npy'
    filepath = os.path.join(models_directory, filename)
    word_vectors, word2ind = word2vec(corpus, word2vec_config)
    np.save(filepath, word_vectors)
    print('saved word vectors as {}!'.format(filepath))


if __name__ == '__main__':
    run()