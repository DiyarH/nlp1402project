import nltk
import torch
from torch.utils.data import Dataset
from torch.nn.utils.sequence import pad_sequence
from transformers import BertTokenizer, BertModel


def get_sentence_length(sample):
    tokenized = nltk.word_tokenize(sample)
    sample_size = len(tokenized)
    return torch.tensor(sample_size)


def get_word_lengths(sample):
    tokenized = nltk.word_tokenize(sample)
    word_lengths = list(map(len, tokenized))
    return torch.tensor(word_lengths)


def get_words(sample, word2idx):
    tokenized = nltk.word_tokenize(sample)
    word_indices = [word2idx[word] for word in tokenized]
    return torch.tensor(word_indices)


def get_word_bigrams(sample, word2idx):
    tokenized = nltk.word_tokenize(sample)
    word_indices = [word2idx[word] for word in tokenized]
    word_bigrams = list(zip(word_indices[1:], word_indices[:-1]))
    return torch.tensor(word_bigrams)


def get_word2vec(sample, word2idx, word2vec):
    tokenized = nltk.word_tokenize(sample)
    word_indices = [word2idx[word] for word in tokenized]
    word_vectors = [word2vec[idx] for idx in word_indices]
    return torch.stack(word_vectors, dim=0)


def get_word2vec_bigrams(sample, word2idx, word2vec):
    tokenized = nltk.word_tokenize(sample)
    word_indices = [word2idx[word] for word in tokenized]
    word_vectors = [word2vec[idx] for idx in word_indices]
    word_vectors_tensor = torch.stack(word_vectors, dim=0)
    word_vector_bigrams = torch.cat(
        (word_vectors_tensor[1:], word_vectors_tensor[:-1]), dim=-1
    )
    return word_vector_bigrams


def get_bert_vectors(sample, tokenizer, weights):
    tokens = tokenizer(sample, return_type="pt").input_ids
    bert_vectors = weights[tokens]
    return bert_vectors


feature_functions = {
    "sentence_length": get_sentence_length,
    "word_length": get_word_lengths,
    "words": get_words,
    "word_bigrams": get_word_bigrams,
    "word2vec": get_word2vec,
    "word2vec_bigrams": get_word2vec_bigrams,
    "bert": get_bert_vectors,
}

def HowTo100MSubtitleFeaturesDataset(Dataset):

    def __init__(self, dataset, feature_names, category2idx):
        if len(feature_names) == 1:
            self.features = extract_features(dataset, feature_names[0])
        else:
            self.features = {}
            for feature_name in feature_names:
                self.features[feature_name] = extract_features(dataset, feature_name)
        
        self.num_classes = len(category2idx)
        self.labels = [category2idx[label] for label in dataset['category']]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def extract_features(dataset, feature_name):
    if feature_name not in feature_functions:
        return None
    feature_function = feature_functions[feature_name]
    if feature_name == "bert":
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        bert_weights = bert_tokenizer.embeddings.word_embeddings.weight
        features = [
            feature_function(sample["text"], bert_tokenizer, bert_weights)
            for sample in dataset
        ]
    elif "word2vec" in feature_name:
        nltk.download("punkt")
        word2ind = load_word2ind()
        if word2ind is None:
            return None
        word2vec = load_word2vec()
        if word2vec is None:
            return None
        features = [
            feature_function(sample["text"], word2ind, word2vec) for sample in dataset
        ]
    elif "length" in feature_name:
        features = [feature_function(sample["text"]) for sample in dataset]
    else:
        nltk.download("punkt")
        word2ind = load_word2ind()
        if word2ind is None:
            return None
        features = [feature_function(sample["text"], word2ind) for sample in dataset]
    if features[0].dim == 0:
        return torch.tensor(features)
    padded_features = pad_sequence(features)
    return padded_features
