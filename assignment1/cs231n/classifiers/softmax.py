from builtins import range
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
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        # i is the image under consideration
        raw_scores = X[i].dot(W)
        max_score = np.max(raw_scores)
        scores = raw_scores - max_score
        correct_class_score = scores[y[i]]
        dW[:, y[i]] -= X[i, :]
        for j in range(num_classes):
            # j is the class
            dW[:, j] +=  X[i, :]*np.exp(scores[j])/np.sum(np.exp(scores))
        loss += -correct_class_score + np.log(np.sum(np.exp(scores)))

    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW+=2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    ytrue_class_prob = np.array([[i, y] for i, y in enumerate(y)])

    scores = np.matmul(X, W)
    scores -= np.max(scores, axis=1).reshape(num_train, 1)

    p = np.exp(scores)
    f = p[ytrue_class_prob[:, 0], ytrue_class_prob[:, 1]]/np.sum(p, axis=1).reshape(1,num_train)

    p_ = -np.log(f)
    loss = np.sum(p_)
    loss /= num_train
    loss += reg*np.sum(W * W)

    p /= np.sum(p, axis=1).reshape(num_train, 1)

    p[ytrue_class_prob[:, 0], ytrue_class_prob[:, 1]] -= 1

    dW = np.matmul(X.T, p)
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
