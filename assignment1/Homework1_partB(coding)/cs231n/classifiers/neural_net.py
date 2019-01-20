from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)/np.sqrt(input_size/2)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)/np.sqrt(hidden_size/2)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    score1 = np.dot(X, W1) + b1  # N by H
    relu = np.maximum(0, score1)  # relu
    score2 = np.dot(relu, W2) + b2  # N by C
    scores = score2
    # b1.reshape(-1,1)
    # b2 = b2.reshape(-1,1)
    # #
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # If the targets are not given then jump out, we're done
    # If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    #     the score for class c on input X[i].
    if y is None:
      return scores
    # #If y is not None, instead return a tuple of:
    # - loss: Loss (data loss and regularization loss) for this batch of training
    #   samples.
    # - grads: Dictionary mapping parameter names to gradients of those parameters
    #   with respect to the loss function; has the same keys as self.params.
    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    num_classes = W2.shape[1]  # 10
    num_train = X.shape[0]  # 500

    # dW = np.zeros(W.shape)
    softscore = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    loss = (-scores[range(num_train), y].sum() + np.sum(np.log(np.sum(np.exp(scores), axis=1))))/ num_train

    loss = loss + 0.5 * reg * np.sum(W1 ** 2) + 0.5 * reg * np.sum(W2**2)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################、

    dsoftscore = softscore
    dsoftscore[np.arange(N), y] -= 1
    dsoftscore /= N

    dW2 = np.dot(relu.T, dsoftscore) + reg * W2
    db2 = np.sum(dsoftscore, axis=0, keepdims=True)

    drelu = np.dot(dsoftscore, W2.T)
    dscore1 = drelu
    dscore1[score1 < 0] = 0

    dW1 = np.dot(X.T, dscore1) + reg * W1
    db1 = np.sum(dscore1, axis=0, keepdims=True)

    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['b1'] = db1
    grads['b2'] = db2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # If the targets are not given then jump out, we're done
    # If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    #     the score for class c on input X[i].

    # #If y is not None, instead return a tuple of:
    # - loss: Loss (data loss and regularization loss) for this batch of training
    #   samples.
    # - grads: Dictionary mapping parameter names to gradients of those parameters
    #   with respect to the loss function; has the same keys as self.params.
    # Compute the loss
    # loss = None
    # num_classes = W2.shape[1]  # 10
    # num_train = X.shape[0]  # 500
    # score1 = np.dot(X, W1) + b1  # N by H
    # relu = np.maximum(0, score1)  # relu
    # score2 = np.dot(relu, W2) + b2  # N by C
    # scores = np.exp(score2) / np.sum(np.exp(score2), axis=1, keepdims=True)
    # correct_score = score2[range(num_train), y]
    #
    # exp_class_score=np.exp(scores)
    # exp_correct_class_score=exp_class_score[np.arange(N),y]
    #
    # loss=-np.log(exp_correct_class_score/np.sum(exp_class_score,axis=1))
    # loss=sum(loss)/N
    #
    # loss+=reg*(np.sum(W2**2)+np.sum(W1**2))
    #
    # # loss = -scores[range(num_train), y].sum() + np.log(np.sum(np.exp(scores), axis=1)).sum()
    # #
    # # loss = loss / num_train + 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
    #
    #
    # if y is None:
    #   return scores
    # dh2 = correct_score / np.sum(correct_score, axis=1,keepdims=True)
    # dh2[np.arange(N),y]-=1
    # dh2/=N
    #
    # dW2=np.dot(score1.T,dh2)
    # dW2+=2*reg*W2
    #
    # db2=np.sum(dh2,axis=0)
    #
    # #layer1
    # dh1=np.dot(dh2,W2.T)
    #
    # dW1X_b1=dh1
    # dW1X_b1[score1<=0]=0
    #
    # dW1=np.dot(X.T,dW1X_b1)
    # dW1 += 2*reg * W1
    #
    # db1=np.sum(dW1X_b1,axis=0)
    # # grads = {}
    # # correct_score = score2[range(num_train),y]
    # # dw2 = np.dot(relu.T, score2[range(num_train),y]) -1 # H by C
    # # dw2/=num_train
    # # db2 = np.sum(score2(range(num_train),y)/num_train, axis=0, keepdims=True)  # (1,C)
    # #
    # # # relu
    # # dscore1 = score2(range(num_train),y)/num_train.dot(W2.T)
    # # dscore1[score1 < 0] = 0  # (N,H)
    # #
    # # # score1
    # # dw1 = np.dot(X.T, dscore1) # (D,H)
    # db1 = np.sum(dh1, axis=0, keepdims=True)  # (1,H)
    #
    # # Add regularization gradient contribution
    # # dW2 += reg * W2
    # # dW1 += reg * W1
    # grads = {}
    # grads['W1'] = dh1
    # grads['W2'] = dh2
    # grads['b1'] = db1
    # grads['b2'] = db2



    # dW1 = np.zeros(W1.shape)
    # dW2 = np.zeros(W2.shape)
    #
    # tmp = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(num_train, 1)
    # tmp[range(num_train), y] -= 1
    #
    # dW2 = np.dot(score1.T, tmp)
    # dW2 = dW2 / num_train + reg * W2
    # b2 = np.sum(tmp,axis=0)
    #
    # dW1 = np.dot(sco.T, tmp)
    # dW1 = dW1 / num_train + reg * W2


    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,batch_size=200,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      randomIndex = np.random.choice(num_train, batch_size, replace=True)
      X_batch = X[randomIndex]
      y_batch = y[randomIndex]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      # W1 = self.params['W1']
      # W2 = self.params['W2']
      # b1 = self.params['b1']
      # b2 = self.params['b2']
      # grads['b1'].reshape(-1,1).T
      # grads['b2'].reshape(-1,1).T

      self.params['W1'] -= grads['W1'] * learning_rate
      self.params['W2'] -= grads['W2'] * learning_rate
      self.params['b1'] -= np.squeeze(grads['b1'] * learning_rate)
      self.params['b2'] -= np.squeeze(grads['b2'] * learning_rate)
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################

    score = self.loss(X)
    y_pred = np.argmax(score, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


