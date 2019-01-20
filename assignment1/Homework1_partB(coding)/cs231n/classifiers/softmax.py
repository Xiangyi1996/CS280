import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  dW = np.zeros_like(W) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  #score = X.dot(W)
  #gt = score[np.arange(num_train), y]  # 500,

  for i in range(num_train):
    score = X[i].dot(W)
    score -= np.max(score)
    loss += np.log(np.sum(np.exp(score)))-score[y[i]]
    dW[:, y[i]] -= X[i]
    for j in range(num_classes):
      dW[:, j] += np.exp(score[j]) / np.exp(score).sum() * X[i]
    # for j in range(num_classes):
    #   if( j != y[i]):
    #     dW[:,j] = np.exp(score[j])/np.sum(np.exp(score)) * X[i]
    #   if( j == y[i]):
    #     dW[:,j] = np.exp(score[j])/np.sum(np.exp(score)) * X[i] - X[i]
  loss = loss/num_train + 0.5*reg*np.sum(W*W)
  dW = dW/num_train + reg * W


  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  loss = 0.0

  num_classes = W.shape[1]  # 10
  num_train = X.shape[0]  # 500
  score = X.dot(W)  # 500*10
  score -= score.max(axis=1).reshape(-1, 1)
  # score.shape #49000 * 10
  dW = np.zeros(W.shape)

  loss = -score[range(num_train), y].sum() + np.log(np.sum(np.exp(score),axis=1)).sum()

  loss = loss / num_train + 0.5 * reg * np.sum(W * W)

  tmp = np.exp(score) / np.sum(np.exp(score), axis=1).reshape(num_train, 1)
  tmp[range(num_train),y]-=1
  dW = np.dot(X.T, tmp)
  dW = dW / num_train + reg * W

  # loss = 0.0
  #
  # num_classes = W.shape[1]  # 10
  # num_train = X.shape[0]  # 500
  # score = X.dot(W)  # 500*10
  # score -= score.max(axis=1).reshape(-1, 1)
  # # score.shape #49000 * 10
  # dW = np.zeros(W.shape)
  #
  # loss = -score[range(num_train), y].sum() + np.log(np.sum(np.exp(score).axis=1)).sum()
  #
  # loss /= num_train
  #
  # np.exp(score)[0]
  # (np.sum(np.exp(score), axis=1).reshape(num_train, 1) - 1)[0]
  # np.exp(score)[0] / (np.sum(np.exp(score), axis=1).reshape(num_train, 1) - 1)[0]
  # dW = np.dot(X.T, (np.exp(score) / np.sum(np.exp(score), axis=1).reshape(num_train, 1)) - 1)
  # dW /= num_train


  return loss, dW

