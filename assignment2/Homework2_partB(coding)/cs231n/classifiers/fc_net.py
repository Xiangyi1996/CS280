from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['b1'] = 0
        self.params['b2'] = 0
        self.params['w1'] = weight_scale * np.random.randn(input_dim, hidden_dim) # D * H
        self.params['w2'] = weight_scale * np.random.randn(hidden_dim, num_classes) # H * C

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        out1,cache1 = affine_forward(X,w1,b1)
        relu_out,relu_cache = relu_forward(out1)
        out2,cache2 = affine_forward(relu_out,w2,b2)
        scores=out2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dout2 = softmax_loss(out2,y)
        reg_loss = 0.5 * self.reg * (np.sum(w2 ** 2) + np.sum(w1 ** 2))
        loss += reg_loss
        drelu_out,dw2,db2 = affine_backward(dout2,cache2)
        dout1 = relu_backward(drelu_out,relu_cache)
        dx,dw1,db1 = affine_backward(dout1,cache1)

        dw1 += self.reg * w1
        dw2 += self.reg * w2


        grads['w1'] = dw1
        grads['w2'] = dw2
        grads['b1'] = db1
        grads['b2'] = db2



        # reg = 0.5
        # input_dim = self.params['W1'].shape[0]
        #
        #
        # loss = -scores[range(3), y].sum() + np.log(np.sum(np.exp(scores), axis=1)).sum()
        # loss = loss/input_dim + 0.5 * reg * np.sum(self.params['W1']**2) + 0.5 * reg * np.sum(self.params['W2']**2)
        # tmp = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(3, 1)
        # tmp[range(3), y] -= 1
        # grads['W2'] = np.dot(relu.T, tmp) # H * C
        # drelu = grads['W2']
        # drelu[score1 < 0] = 0
        # cache = X,self.params['W1'],self.params['b1']
        # grads['W1'] = layers.affine_backward(drelu, cache)
        # scores -= scores.max(axis=1).reshape(-1, 1)
        #
        # grads['w2'] = np.zeros(self.params['w2'].shape)
        #
        # loss = -scores[range(input_dim), y].sum() + np.log(np.sum(np.exp(scores), axis=1)).sum()
        #
        # loss = loss / input_dim + 0.5 * reg * np.sum(self.params['w2'] * self.params['w2'])
        #
        # tmp = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(input_dim, 1)
        # tmp[range(input_dim), y] -= 1
        # grads['w2'] = np.dot(X.T, tmp)
        # grads['w2'] = grads['w2'] / input_dim + reg * self.params['w2']
        #
        # grads['w1'] = X.T.dot(dout)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout >0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims) #3
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        for layer in range(self.num_layers): # layer = 0,1,2
            if layer ==0:
                layer_dim = (input_dim,hidden_dims[0])
            elif layer == self.num_layers-1:
                layer_dim = (hidden_dims[layer-1],num_classes)
            else:
                layer_dim = (hidden_dims[layer-1],hidden_dims[layer])
            self.params['w%d'%(layer+1)] = weight_scale * np.random.randn(layer_dim[0],layer_dim[1])
            self.params['b%d'%(layer+1)] = np.zeros(layer_dim[1])

            if self.normalization=='batchnorm' and layer!=self.num_layers-1:
                self.params['gamma%d' % (layer + 1)] = np.ones(layer_dim[1])
                self.params['beta%d' % (layer + 1)] = np.zeros(layer_dim[1])



        # w1, b1, gamma1, beta1 = self.params['w1'], self.params['b1'], self.params['gamma1'], self.params['beta1']
        # w2, b2, gamma2, beta2 = self.params['w2'], self.params['b2'], self.params['gamma2'], self.params['beta2']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        # for k, v in self.params.items():
        #     self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        aff_cache_list=[]
        batch_cache_list=[]
        relu_cache_list=[]
        drop_cache_list=[]

        outi=X

        for layer in range(self.num_layers):# layer 0 1 2
            wi,bi = self.params['w%d'%(layer+1)],self.params['b%d'%(layer+1)]
            #print(wi)

            #affine
            outi,cachei = affine_forward(outi,wi,bi)

            aff_cache_list.append(cachei)

            #batchnormal

            if self.normalization=='batchnorm' and layer!=(self.num_layers-1):
                gammai = self.params['gamma%d' % (layer + 1)]
                #print('gamma',gammai.shape)
                betai = self.params['beta%d' % (layer + 1)]
                outi, cachei = batchnorm_forward(outi,gammai,betai,self.bn_params[layer])
                batch_cache_list.append(cachei)
                #print('batch_list',batch_cache_list.__len__())

            #relu
            outi,cachei = relu_forward(outi)
            relu_cache_list.append(cachei)
            #print('relu_list',relu_cache_list.__len__())

            #dropout
            if self.use_dropout:
                outi,cachei = dropout_forward(outi,self.dropout_param)
                drop_cache_list.append(cachei)
            scores = outi
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #softmax
        loss,dout = softmax_loss(scores,y)
        #print('score',scores.shape)
        #print('y',y.shape)
        reg_loss = 0
        #print('dout', dout.shape)
        for layer in range(self.num_layers):
            wi,bi = self.params['w%d'%(layer+1)],self.params['b%d'%(layer+1)]

            reg_loss += 0.5 * self.reg * np.sum(wi**2)
        loss += reg_loss

        for layer in list(range(self.num_layers, 0, -1)):
            #print(layer - 1)
            # #affine
            # dout, dw, db = affine_backward(dout, aff_cache_list[layer-1])
            # a = self.params['w%d'%(layer-1)]
            #
            # print(dw.shape,a.shape)

            # dw += self.reg * self.params['w%d'%(layer)]
            #
            # grads['dw%d'%(layer)],grads['db%d'%(layer)] = dw,db

            #dropout
            if self.use_dropout:
                dout = dropout_backward(dout,drop_cache_list[layer-1])
                #print('dout',dout.shape)
            # relu

            dout = relu_backward(dout, relu_cache_list[layer-1])
            #batchnormal
            if self.normalization =='batchnorm' and layer != self.num_layers:
                #print(batch_cache_list.__len__())
                dout, dgamma, dbeta = batchnorm_backward(dout, batch_cache_list[layer-1])
                grads['gamma%d' % (layer)] = dgamma
                grads['beta%d' % (layer)] = dbeta

            #affine
            dout, dw, db = affine_backward(dout, aff_cache_list[layer-1])
            dw += self.reg * self.params['w%d' % (layer)]
            grads['w%d' % (layer)], grads['b%d' % (layer)] = dw, db



        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
