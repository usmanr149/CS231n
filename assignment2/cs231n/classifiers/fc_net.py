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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)

        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # first layer activation
        # print('X.shape: ', X.shape)
        # print("self.params['W1'].shape: ", self.params['W1'].shape)
        # print("self.params['b1'].shape: ", self.params['b1'].shape)

        # a = np.matmul(X.reshape(self.input_dim, self.num_classes), self.params['W1']) + self.params['b1']
        # a = affine_forward(X, self.params['W1'], self.params['b1'])

        N = X.shape[0]
        X = np.reshape(X, [N, -1])  # Flatten images.


        a = np.matmul(X, self.params['W1']) + self.params['b1']
        a[a < 0] = 0
        b = a

        # second layer activation
        scores = np.matmul(b, self.params['W2']) + self.params['b2']
        # scores = affine_forward(b, self.params['W2'], self.params['b2'])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        ytrue_class_prob = np.array([[i, y] for i, y in enumerate(y)])
        # scores -= np.max(scores, axis=1).reshape(N, 1)
        # print(scores)
        d = np.exp(scores)
        f = d[ytrue_class_prob[:, 0], ytrue_class_prob[:, 1]] / np.sum(d, axis=1).reshape(1, N)

        p_ = -np.log(f)
        loss = np.sum(p_)
        loss /= N
        loss += 0.5 * self.reg * (np.sum(self.params['W1'] * self.params['W1']) + np.sum(self.params['W2'] * self.params['W2']))
        # print('loss: ', loss)

        # N * C
        grad_L_wrt_c = d / np.sum(d, axis=1, keepdims=True)
        # print('grad_L_wrt_c: ', grad_L_wrt_c)

        grad_L_wrt_c[ytrue_class_prob[:, 0], ytrue_class_prob[:, 1]] -= 1
        grad_L_wrt_c /= N
        # print('grad_L_wrt_c: ', grad_L_wrt_c)
        # N * C
        grad_L_wrt_W2 = b.T.dot(grad_L_wrt_c)
        # print('grad_L_wrt_W2: ', grad_L_wrt_W2)

        # N * h
        grad_L_wrt_b = grad_L_wrt_c.dot(self.params['W2'].T)
        # print('grad_L_wrt_b: ', grad_L_wrt_b)

        grad_L_wrt_b2 = np.sum(grad_L_wrt_c, axis=0)
        # print('grad_L_wrt_b2: ', grad_L_wrt_b2.shape)

        grad_L_wrt_a = np.where(a <= 0, 0, 1) * grad_L_wrt_b
        # print('grad_L_wrt_a: ', grad_L_wrt_a)

        grad_L_wrt_W1 = X.T.dot(grad_L_wrt_a)
        # print('grad_L_wrt_W1: ', grad_L_wrt_W1)

        grad_L_wrt_X = grad_L_wrt_a.dot(self.params['W1'].T)
        # print('grad_L_wrt_X: ', grad_L_wrt_X)

        grad_L_wrt_b1 = np.sum(grad_L_wrt_a, axis=0)
        # print('grad_L_wrt_b1: ', grad_L_wrt_b1.shape)

        # print('grad_L_wrt_W1.shape: ', grad_L_wrt_W1.shape)
        # print('grad_L_wrt_W2.shape: ', grad_L_wrt_W2.shape)

        grads['W1'] = grad_L_wrt_W1 + self.reg * self.params['W1']
        grads['b1'] = grad_L_wrt_b1
        grads['W2'] = grad_L_wrt_W2 + self.reg * self.params['W2']
        grads['b2'] = grad_L_wrt_b2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.cache = {}

        # first layer weights
        self.params['W1'] = np.random.normal(0, weight_scale, [input_dim, hidden_dims[0]])
        self.params['b1'] = np.zeros([hidden_dims[0]])
        if self.normalization=='batchnorm':
            self.params['gamma0'] = np.random.randn(hidden_dims[0])
            self.params['beta0'] = np.random.randn(hidden_dims[0])

        for i in range(1, self.num_layers - 1):
            self.params['W' + str(i+1)] = np.random.normal(0, weight_scale, [hidden_dims[i - 1], hidden_dims[i]])
            self.params['b' + str(i+1)] = np.zeros([hidden_dims[i]])
            if self.normalization == 'batchnorm':
                self.params['gamma' + str(i)] = np.random.randn(hidden_dims[i])
                self.params['beta' + str(i)] = np.random.randn(hidden_dims[i])

        # the final layer
        self.params['W' + str(self.num_layers)] = np.random.normal(0, weight_scale,
                                                                   [hidden_dims[self.num_layers - 2],
                                                                    num_classes])
        self.params['b' + str(self.num_layers)] = np.zeros([num_classes])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
            self.bn_params = [{'mode': 'train', i:{'eps': None, 'momentum': None,
                                                   'running_mean': None, 'running_var': None,
                                  'cache': {}}} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        N = X.shape[0]
        X = np.reshape(X, [N, -1])  # Flatten images.

        # first layer
        self.cache['H1'] = X.dot(self.params['W1']) + self.params['b1']
        if self.normalization == 'batchnorm':
            self.cache['H1'], self.bn_params[0]['cache'] = batchnorm_forward(self.cache['H1'], self.params['gamma0'],
                                                                            self.params['beta0'], self.bn_params[0])
        self.cache['A1'] = np.maximum(0, self.cache['H1'])
        # self.params['H1'][self.params['H1'] < 0] = 0

        #Intermediate hidden laters
        for i in range(1, self.num_layers - 1):
            self.cache['H' + str(i+1)] = self.cache['A' + str(i)].dot(self.params['W' + str(i+1)]) + \
                                              self.params['b' + str(i+1)]
            if self.normalization=='batchnorm':
                self.cache['H' + str(i+1)], self.bn_params[i]['cache'] = batchnorm_forward(self.cache['H' + str(i+1)], \
                            self.params['gamma' + str(i)], self.params['beta' + str(i)], self.bn_params[i])

            self.cache['A' + str(i + 1)] = np.maximum(0, self.cache['H' + str(i+1)])

        # output layer
        self.cache['H' + str(self.num_layers)] = \
            self.cache['A' + str(self.num_layers - 1)].dot(self.params['W' + str(self.num_layers)]) \
            + self.params['b' + str(self.num_layers)]
        self.cache['A' + str(self.num_layers)] = self.cache['H' + str(self.num_layers)]
        scores = self.cache['A' + str(self.num_layers)]
        # self.params['A' + str(self.num_layers)] = np.maximum(0, self.params['H' + str(self.num_layers)])


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        d = np.exp(self.cache['A' + str(self.num_layers)])
        f = d[np.arange(N), y] / np.sum(d, axis=1).reshape(1, N)

        p_ = -np.log(f)
        loss = np.sum(p_)
        loss /= N
        for i in range(self.num_layers):
            loss += 0.5 * self.reg * (np.sum(self.params['W' + str(i + 1)] * self.params['W' + str(i + 1)]) )

        cache = {}
        cache['A' + str(self.num_layers)] = d / np.sum(d, axis=1, keepdims=True)

        cache['A' + str(self.num_layers)][np.arange(N), y] -= 1

        cache['A' + str(self.num_layers)] /= N
        cache['H' + str(self.num_layers)] = cache['A' + str(self.num_layers)]

        # first backward pass
        grads['W' + str(self.num_layers)] = \
            self.cache['A' + str(self.num_layers - 1)].T.dot(cache['H' + str(self.num_layers)]) + \
            self.reg*self.params['W' + str(self.num_layers)]

        grads['b' + str(self.num_layers)] = np.sum(cache['H' + str(self.num_layers)], axis=0)

        for i in range(self.num_layers - 1, 0, -1):
            if i != 1:
                cache['A' + str(i)] = cache['H' + str(i + 1)].dot(self.params['W' + str(i + 1)].T)
                cache['H' + str(i)] = cache['A' + str(i)] * np.where(self.cache['A' + str(i)] <= 0, 0, 1)
                if self.normalization == 'batchnorm':
                    cache['H' + str(i)], grads['gamma' + str(i - 1)], grads['beta' + str(i - 1)] = \
                        batchnorm_backward_alt(cache['H' + str(i)], self.bn_params[i-1]['cache'])

                grads['W' + str(i)] = self.cache['A' + str(i - 1)].T.dot(cache['H' + str(i)]) + \
                                      self.reg * self.params['W' + str(i)]
                grads['b' + str(i)] = np.sum(cache['H' + str(i)], axis=0)
            else:
                cache['A' + str(i)] = cache['H' + str(i + 1)].dot(self.params['W' + str(i + 1)].T)
                cache['H' + str(i)] = cache['A' + str(i)] * np.where(self.cache['A' + str(i)] <= 0, 0, 1)
                if self.normalization == 'batchnorm':
                    cache['H' + str(i)], grads['gamma' + str(i - 1)], grads['beta' + str(i - 1)] = \
                        batchnorm_backward_alt(cache['H' + str(i)], self.bn_params[i - 1]['cache'])

                grads['W' + str(i)] = X.T.dot(cache['H' + str(i)]) + self.reg*self.params['W' + str(i)]
                grads['b' + str(i)] = np.sum(cache['H' + str(i)], axis=0)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
