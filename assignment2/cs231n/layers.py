from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_inputs = x.shape[0]
    input_shape = x.shape[1:]
    output_dim = b.shape[0]

    # reshaping beuse input is an RGB image from CIFAR-10 dataset
    out = x.reshape(num_inputs, np.prod(input_shape)).dot(w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_inputs = x.shape[0]
    input_shape = x.shape[1:]
    output_dim = b.shape[0]

    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    dw = x.reshape(num_inputs, np.prod(input_shape)).T.dot(dout)
    db = np.sum(dout, axis = 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(x, 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = np.where(x>=0, 1, 0)
    dx = dout*dx

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        sample_mean = x.mean(axis=0)

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean

        sample_var = x.var(axis=0)
        running_var = momentum * running_var + (1 - momentum) * sample_var

        x_norm = (x - sample_mean)/np.sqrt(sample_var + eps)
        out = gamma*x_norm + beta

        cache = {}
        cache['gamma'] = gamma
        cache['beta'] = beta
        cache['x_norm'] = x_norm
        cache['x'] = x
        cache['sample_mean'] = sample_mean
        cache['sample_var'] = sample_var
        cache['eps'] = eps
        cache['N'], cache['D'] = N, D

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # out = (x - bn_param['running_mean'])/np.sqrt(bn_param['running_var'] + eps)
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dgamma = np.sum(dout*cache['x_norm'], axis = 0)
    dbeta = np.sum(dout, axis = 0)

    x = cache['x']
    mu =  cache['sample_mean']
    sigma = cache['sample_var']
    eps = cache['eps']
    gamma = cache['gamma']
    beta = cache['beta']
    N = cache['N']

    b = x - mu
    c = sigma + eps
    d = np.sqrt(c)
    e = 1/d
    f = b*e
    g = gamma*f
    y = g + beta

    dx = -(2/N)*(x-mu)*np.sum( dout*gamma*b, axis = 0)*(1/d**2)*(1/2)*(1/np.sqrt(c))  - (1/N)*np.sum( dout*gamma*e , axis = 0) + dout*gamma*e

    # dx_hat = dout*cache['gamma']
    #
    # dsigma = (-1./2) * (cache['sample_var'] + cache['eps']) ** (-3. / 2) * \
    #          np.sum(dx_hat * (cache['x'] - cache['sample_mean']), axis=0)
    #
    # dmu = np.sum(dx_hat * (-1. / (np.sqrt(cache['sample_var'] + cache['eps']))), axis=0) + \
    #       np.sum(2. * (cache['x'] - cache['sample_mean']), axis=0) / cache['N']
    #
    # dx = dx_hat * (1. / (np.sqrt(cache['sample_var'] + cache['eps']))) + \
    #      2. * dsigma * (cache['x'] - cache['sample_mean']) / cache['N'] + \
    #      dmu * (1. / cache['N'])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dgamma = np.sum(dout * cache['x_norm'], axis=0)
    dbeta = np.sum(dout, axis=0)

    v = cache['sample_var'] + cache['eps']

    dx = cache['gamma'] * (dout / v ** (1 / 2) - \
                           (1 / ((cache['N'] * v ** (3 / 2)))) * (np.sum(dout * v, axis=0) + (cache['x'] - cache['sample_mean'])
                                                                  * np.sum(dout * (cache['x'] - cache['sample_mean']),axis=0) ) )

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    sample_mean = x.mean(axis=1)

    sample_var = x.var(axis=1)

    x_norm = ((x.T - sample_mean) / np.sqrt(sample_var + eps)).T
    out = gamma * x_norm + beta

    cache = {}
    cache['gamma'] = gamma
    cache['beta'] = beta
    cache['x_norm'] = x_norm
    cache['x'] = x
    cache['sample_mean'] = sample_mean
    cache['sample_var'] = sample_var
    cache['eps'] = eps
    cache['N'], cache['D'] = x.shape

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for k, v in cache.items():
        try:
            print(k, v.shape)
            print(v)
        except:
            pass

    dgamma = np.sum(dout * cache['x_norm'], axis=0)
    dbeta = np.sum(dout, axis=0)

    v = cache['sample_var'] + cache['eps']

    dx = ( (cache['gamma'] * dout).T / v ** (1 / 2)  ) - ((1 / ((cache['D'] * v ** (3 / 2))))*(\
    np.sum(dout * cache['gamma'], axis=1)*v + (cache['x'].T - cache['sample_mean'])* \
    np.sum( (dout*cache['gamma']).T * (cache['x'].T - cache['sample_mean']), axis=0)) )

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx.T, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p) / p # dropout mask.
        out = x * mask  # drop!

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        dx = dout * mask
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    S, P = conv_param['stride'], conv_param['pad']
    output_d = (W - HH + 2*P)/S + 1
    # To make sure that dimensions work out
    assert output_d == int(output_d)

    output_d = int(output_d)

    output_volume = (N, F, output_d, output_d)

    V = np.zeros(output_volume)

    X = np.zeros((N, C, H+2*P, W+2*P))

    # add padding to x
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            X[i][j] = np.pad(x[i][j], P, mode='constant')

    for n in range(N):
        for f in range(F):
            for i in range(output_d):
                for j in range(output_d):
                    # print(X[n, f, S*i: S*i + HH, S*j: S*j + WW] * w[f])
                    V[n, f, i, j] += np.sum(X[n, :, S*i: S*i + HH, S*j: S*j + WW] * w[f]) + b[f]

    out = V

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW  = w.shape
    S, P = conv_param['stride'], conv_param['pad']
    output_d = (W - HH + 2 * P) / S + 1
    # To make sure that dimensions work out
    if output_d == int(output_d):
        output_d = int(output_d)
    else:
        return

    X = np.zeros((N, C, H + 2, W + 2))

    # add padding to x
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            X[i][j] = np.pad(x[i][j], 1, mode='constant')

    # print('X: ', np.around(X, decimals=4))

    # Number of b is equivalent to the number of filters
    db = np.zeros(F)
    for f in range(F):
        for n in range(N):
            db[f] += np.sum(dout[n, f, : ,:])

    dw = np.zeros((F, C, HH, WW))
    for c in range(C):
        for f in range(F):
            for i in range(WW):
                matrix_elements_i = np.array( [[S*i_ + i] * output_d for i_ in range(output_d)] )
                for j in range(HH):
                    matrix_elements_j = np.array( [[S*i_ + j]*output_d for i_ in range(output_d)] ).T
                    dw[f, c, i, j] += np.sum(dout[:, f, :, :] * X[:, c, matrix_elements_i, matrix_elements_j])

    dx = np.zeros((N, C, H, W))
    for f in range(F):
        for c in range(C):
            for n in range(N):
                for w1 in range(output_d):
                    for w2 in range(output_d):
                        for i in range( max(0, w1*S - P), min(W, w1*S + WW - P) ):
                            for j in range( max(0, w2*S - P), min(H, w2*S + HH - P) ):
                                dx[n, c, i, j]+=np.sum(dout[n, f, w1, w2] * w[f, c, i + P - w1*S, j + P - w2*S])


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    pool_height, pool_width, S = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    H_pool = (H - pool_height)//S + 1
    W_pool = (W - pool_width)//S + 1
    out = np.zeros((N, C, H_pool, W_pool))

    for n in range(N):
        for c in range(C):
            for i in range(H_pool):
                pool_i = np.array([[i_]*pool_height for i_ in range(i*S, i*S + pool_height)])
                for j in range(W_pool):
                    pool_j = np.array([[j_]*pool_width for j_ in range(j*S, j*S + pool_width)]).T
                    out[n,c,i,j] = np.max(x[n,c, pool_i, pool_j])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache

    N, C, H, W = x.shape
    pool_height, pool_width, S = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    H_pool = (H - pool_height) // S + 1
    W_pool = (W - pool_width) // S + 1

    dx = np.zeros((N, C, H, W))

    for n in range(N):
        for c in range(C):
            for i in range(H_pool):
                pool_i = np.array([[i_] * pool_height for i_ in range(i * S, i * S + pool_height)])
                for j in range(W_pool):
                    pool_j = np.array([[j_] * pool_width for j_ in range(j * S, j * S + pool_width)]).T
                    max_val = np.amax(x[n,c, pool_i, pool_j])
                    for h in range(i * S, i * S + pool_height):
                        for w in range(j * S, j * S + pool_width):
                            dx[n, c, h, w] += dout[n, c, i, j] * int(x[n, c, h, w] >= max_val)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    x = x.reshape(N*H*W, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)

    out = out.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    dout = dout.reshape(N*H*W,C)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)

    dx = dx.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    #
    # x = x.reshape(N, G, C//G, H, W)
    # x_norm = np.zeros((N, G, C//G, H, W))
    # sample_mean = np.zeros((N, G, C // G, H, W))
    # sample_var = np.zeros((N, G, C // G, H, W))
    #
    # for n in range(N):
    #     for g in range(G):
    #         for c in range(C//G):
    #             sample_mean[n,g,c] = x[n,g,c].mean(axis=0, keepdims=True)
    #
    #             sample_var[n,g,c] = x[n,g,c].var(axis=0, keepdims=True)
    #
    #             x_norm[n,g,c] += ((x[n,g,c] - sample_mean[n,g,c]) / np.sqrt(sample_var[n,g,c] + eps))
    #
    # x = x_norm.reshape(N,C,H,W)
    # sample_mean = sample_mean.reshape(N,C,H,W)
    # sample_var = sample_var.reshape(N,C,H,W)
    #
    # out = gamma * x + beta

    x = x.reshape(N, G, C // G, H*W)
    x_norm = np.zeros((N, G, C // G, H*W))
    sample_mean = np.zeros((N, G, C // G, H*W))
    sample_var = np.zeros((N, G, C // G, H*W))

    # layer norm is acros rows
    for n in range(N):
        for g in range(G):
            for c in range(C // G):
                sample_mean[n, g, c] = x[n, g, c].mean(axis=0, keepdims=True)
                sample_var[n, g, c] = x[n, g, c].var(axis=0, keepdims=True)
                x_norm[n, g, c] += ((x[n, g, c] - sample_mean[n, g, c]) / np.sqrt(sample_var[n, g, c] + eps))

    x = x.reshape(N, C, H, W)
    x_norm = x_norm.reshape(N, C, H, W)
    sample_mean = sample_mean.reshape(N, C, H, W)
    sample_var = sample_var.reshape(N, C, H, W)

    out = gamma * x_norm + beta

    cache = {}
    cache['gamma'] = gamma
    cache['beta'] = beta
    cache['x_norm'] = x_norm
    cache['x'] = x
    cache['sample_mean'] = sample_mean
    cache['sample_var'] = sample_var
    cache['eps'] = eps
    cache['group'] = G

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # for n in range(N):
    #     for c in range(C):
    #         # for cache split everythin
    #
    #         _, dgamma[0, c, 0, 0], dbeta[0, c, 0, 0] = layernorm_backward(dout[n, c], )

    gamma, beta, x_norm, x, sample_mean, sample_var, eps, G = cache['gamma'], cache['beta'], cache['x_norm'], \
                                                              cache['x'], cache['sample_mean'], cache['sample_var'], \
                                                              cache['eps'], cache['group']

    N, C, H, W = x.shape

    # dout = dout.reshape(N, G, C // G, H, W)
    dx = np.zeros((N, G, C // G, H, W))
    dgamma = np.zeros(gamma.shape)
    dbeta = np.zeros(beta.shape)

    dx = np.zeros((N, C, H * W))
    x = x.reshape(N, C, H * W)
    x_norm = x_norm.reshape(N, C, H * W)
    sample_mean = sample_mean.reshape(N, C, H * W)
    dout = dout.reshape(N, C, H * W)
    sample_var = sample_var.reshape(N, C, H * W)
    v = sample_var + cache['eps']
    D = H*W
    step = C // G

    for n in range(N):
        dgamma[0, :, 0, 0] += np.sum(
            dout[n, :] * x_norm[n, :], axis=1)
        dbeta[0, :, 0, 0] += np.sum(dout[n, :], axis=1)
        for s in range(C):
            for d in range(D):
                dx[n, s, d] = (gamma[0, s, 0, 0] * dout[n, s, d]) / v[n, s, d] ** (1 / 2) - (
                            (1 / ((D * v[n, s, d] ** (3 / 2)))) * ( \
                                np.sum(
                                    gamma[0, s, 0, 0] * dout[n, s, :]) *
                                v[n, s, d] + (x[n, s, d] - sample_mean[n, s, d]) * \
                                np.sum((dout[n, s, :] * gamma[0, s, 0, 0]) * (
                                                   x[n, s, :] - sample_mean[n, s, d]))))

    dx = dx.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
