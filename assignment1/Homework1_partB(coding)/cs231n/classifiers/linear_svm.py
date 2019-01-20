import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    for j in range(num_classes):
      margin = scores[j] - scores[y[i]] + 1  # note delta = 1
      if j!= y[i]:
        if scores[y[i]] >= scores[j] + 1:
          loss+=0
        else:
          dW[:,j]+=X[i] #给导数的第j列带来xi的贡献
          dW[:,y[i]]+=(-X[i])
          loss += margin

  #L = sum(margin[margin > 0])
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5*reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0

  num_classes = W.shape[1]  # 10
  num_train = X.shape[0]  # 500
  score = X.dot(W)  # 500*10
  dW = np.zeros_like(W)

  # GT
  gt = score[np.arange(num_train), y] #500,

  gt = gt.reshape(-1, 1)
  gt = np.tile(gt, 10)# 500*10

  margin = score - gt + 1.0

  loss += np.sum(margin[(margin!=1) & (margin >0)])
  loss /= num_train

  loss+=0.5*reg*np.sum(W*W)



  # margin[(margin == 1)] = -1
  # margin[(margin != 1) & (margin != -1)] = 1.0

  # margin_row= np.sum(margin,axis=1)
  # margin[np.arange(num_train),y]=-margin_row.T

  margin[np.arange(num_train), y] = 0

  margin[(margin > 0)] = 1
  margin[margin <= 0] = 0
  row = np.sum(margin, axis=1)
  margin[np.arange(num_train), y] = -row.T
  dW = np.dot(X.T, margin)

  dW /= num_train
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


