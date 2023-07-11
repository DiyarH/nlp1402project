# import numpy as np
import torch
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        tmp = torch.max(x, dim=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = torch.exp(x)
        tmp = torch.sum(x, dim=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # Vector
        tmp = torch.max(x)
        x -= tmp
        x = torch.exp(x)
        tmp = torch.sum(x)
        x /= tmp

    assert x.shape == orig_shape
    return x


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    s = 1 / (1 + torch.exp(-x))

    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec, outsideWordIdx, outsideVectors, vocabulary
):
    """Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models. For those unfamiliar with numpy notation, note
    that a numpy ndarray with a shape of (x, ) is a one-dimensional array, which
    you can effectively treat as a vector with length x.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length)
                    for all words in vocab (tranpose of U in the pdf handout)
    vocabulary -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length)
                    (dJ / dU)
    """

    p_all = softmax(torch.transpose(outsideVectors, 0, 1) @ centerWordVec)
    p_o = p_all[outsideWordIdx]
    loss = -torch.log(p_o)
    u_o = outsideVectors[outsideWordIdx]
    gradCenterVec = torch.sum(outsideVectors * torch.unsqueeze(p_all, 1), dim=0) - u_o
    gradOutsideVecs = torch.broadcast_to(
        centerWordVec, outsideVectors.shape
    ) * torch.unsqueeze(p_all, 1)
    gradOutsideVecs[outsideWordIdx] -= centerWordVec

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, vocabulary, K):
    """Samples K indexes which are not the outsideWordIdx"""

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = random.randint(0, len(vocabulary) - 1)
        while newidx == outsideWordIdx:
            newidx = random.randint(0, len(vocabulary) - 1)
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec, outsideWordIdx, outsideVectors, vocabulary, K=10
):
    """Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    negSampleWordIndices = getNegativeSamples(outsideWordIdx, vocabulary, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    unique_negative_indices, unique_negative_counts = torch.unique(
        torch.tensor(negSampleWordIndices), return_counts=True
    )
    u_o = outsideVectors[outsideWordIdx]
    u_negatives = outsideVectors[unique_negative_indices]
    positive_similarity = torch.unsqueeze(u_o, 0) @ centerWordVec
    negative_similarities = -torch.transpose(u_negatives, 0, 1) @ centerWordVec
    positive_sigmoid = sigmoid(positive_similarity)
    negative_sigmoids = sigmoid(negative_similarities)
    positive_loss = -torch.log(positive_sigmoid)
    negative_losses = -torch.log(negative_sigmoids)
    loss = positive_loss + torch.sum(negative_losses * unique_negative_counts)
    gradCenterVec = (positive_sigmoid - 1) * u_o
    gradCenterVec -= torch.sum(
        torch.unsqueeze(unique_negative_counts * (negative_sigmoids - 1), 1)
        * torch.transpose(u_negatives, 0, 1),
        axis=0,
    )
    gradOutsideVecs = torch.zeros_like(outsideVectors)
    gradOutsideVecs[unique_negative_indices] = torch.unsqueeze(
        unique_negative_counts * (1 - negative_sigmoids), 1
    ) * torch.broadcast_to(centerWordVec, u_negatives.shape)
    gradOutsideVecs[outsideWordIdx] = (positive_sigmoid - 1) * centerWordVec

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(
    currentCenterWord,
    windowSize,
    outsideWords,
    word2Ind,
    centerWordVectors,
    outsideVectors,
    dataset,
    word2vecLossAndGradient=naiveSoftmaxLossAndGradient,
):
    """Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape
                        (num words in vocab, word vector length)
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape
                        (num words in vocab, word vector length)
                        for all words in vocab (transpose of U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vector
                     in shape (num words in vocab, word vector length)
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length)
                    (dJ / dU)
    """

    loss = 0.0
    gradCenterVecs = torch.zeros(centerWordVectors.shape).to(device)
    gradOutsideVectors = torch.zeros(outsideVectors.shape).to(device)

    center_word_index = word2Ind[currentCenterWord]
    center_word_vec = centerWordVectors[center_word_index]
    outside_words_indices = [word2Ind[word] for word in outsideWords]
    for outside_word_index in outside_words_indices:
        loss_and_gradient = word2vecLossAndGradient(
            center_word_vec, outside_word_index, outsideVectors, word2Ind
        )
        (
            current_loss,
            current_grad_center_vec,
            current_grad_outside_vectors,
        ) = loss_and_gradient
        loss += current_loss
        gradCenterVecs[center_word_index] += current_grad_center_vec
        gradOutsideVectors += current_grad_outside_vectors

    return loss, gradCenterVecs, gradOutsideVectors


def get_random_context(dataset, window_size):
    document = random.choice(dataset)
    document_length = len(document)
    center_idx = random.randint(window_size, document_length - 1 - window_size)
    center_word = document[center_idx]
    context = (
        document[center_idx - window_size : center_idx]
        + document[center_idx + 1 : center_idx + window_size + 1]
    )
    return center_word, context


def word2vec_sgd_wrapper(
    word2vecModel,
    word2Ind,
    wordVectors,
    dataset,
    windowSize,
    word2vecLossAndGradient=naiveSoftmaxLossAndGradient,
):
    batchsize = 50
    loss = 0.0
    grad = torch.zeros(wordVectors.shape).to(device)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[: int(N / 2), :]
    outsideVectors = wordVectors[int(N / 2) :, :]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = get_random_context(dataset, windowSize1)

        c, gin, gout = word2vecModel(
            centerWord,
            windowSize1,
            context,
            word2Ind,
            centerWordVectors,
            outsideVectors,
            dataset,
            word2vecLossAndGradient,
        )
        loss += c / batchsize
        grad[: int(N / 2), :] += gin / batchsize
        grad[int(N / 2) :, :] += gout / batchsize

    return loss, grad
