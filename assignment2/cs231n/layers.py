from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

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
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N=x.shape[0]
    X_data=x.reshape((N,-1))
    out=X_data.dot(w)+b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

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
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N=x.shape[0]
    M=w.shape[1]
    db=np.sum(dout,axis=0)
    dx=dout.dot(w.transpose())
    dx=dx.reshape(x.shape)
    dw=x.reshape((N,-1)).transpose().dot(dout)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out=x*(x>0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx=dout*(x>0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N=x.shape[0]
    C=x.shape[1]
    S=x.copy()    
    S_exp=np.exp(S)
    S_exp_sum=np.sum(S_exp,axis=1)
    x_index=np.arange(0,N)
    syi=S_exp[x_index,y]
    s=syi/S_exp_sum
    s_loss=-np.log(s)
    loss=np.sum(s_loss)/N

    grad_L=1
    grad_L/=N
    grad_s_loss=np.ones((N,1))*grad_L
    grad_s=grad_s_loss*(-1/s.reshape((-1,1)))
    # print(grad_s)
    grad_S_exp=-syi.reshape((-1,1)).dot(np.ones((1,C)))
    temp=S_exp_sum-syi
    grad_S_exp[x_index,y]=temp
    temp=S_exp_sum*S_exp_sum
    grad_S_exp=grad_S_exp/temp.reshape((-1,1))
    grad_S_exp*=grad_s
    grad_S=grad_S_exp*S_exp
    dx=grad_S

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

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
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
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

        m=np.mean(x,axis=0)
        v=np.var(x,axis=0)
        y=(x-m)/np.sqrt(v+eps)
        out=gamma*y+beta
        cache=(x,gamma,beta,y,eps)
        running_mean=running_mean*momentum+(1-momentum)*m
        running_var=running_var*momentum+(1-momentum)*v
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out=(x-running_mean)/np.sqrt(running_var+eps)
        out=gamma*out+beta
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

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
    x,gamma,beta,y,eps=cache
    N=x.shape[0]
    m=np.mean(x,axis=0)
    v=np.var(x,axis=0)
    dbeta=np.sum(dout,axis=0)
    dgamma=np.sum(dout*y,axis=0)
    dy=gamma*dout
    dm1=np.sum(-1/(np.sqrt(v+eps))*dy,axis=0)
    
    
    dx1=1/(np.sqrt(v+eps))*dy
    dv=np.sum(dy*(-0.5*(x-m)*(v+eps)**(-1.5)),axis=0)
    #dm
    # print(m-1/N*np.sum(x,axis=0))
    # dm2=2*(m-1/N*np.sum(x,axis=0))*dv
    # print(dm2)
    # dm=dm1+dm2
    dm=dm1
    #dx2
    # dx2=((2*(N+1)/(N*N))*x+2/(N*N)*(np.sum(x,axis=0)-x))*dv
    dx2=2/N*(x-m)*dv
    #dx3 mean
    dx3=dm*np.ones(x.shape)/N
    dx=dx1+dx2+dx3
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

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

    x,gamma,beta,y,eps=cache
    N=x.shape[0]
    D=x.shape[1]
    m=np.mean(x,axis=0)
    v=np.var(x,axis=0)
    sigma=np.sqrt(v+eps)
    dbeta=np.sum(dout,axis=0)
    dgamma=np.sum(dout*y,axis=0)
    dy=gamma*dout
    dx=(-1/N*(np.ones(x.shape)*np.sum(dy,axis=0)+y*np.sum(y*dy,axis=0))+dy)/sigma
    # for col in range(D):
    #   d1=y[:,col].reshape((-1,1))
    #   d2=y[:,col].reshape((1,-1))
    #   if dx is None:
    #     dx=(-(d1.dot(d2)+1)/(N*sigma[col])+np.identity(N)*1/sigma[col]).dot(dy[:,col].reshape((-1,1)))
    #   else:
    #     dx=np.column_stack((dx,(-(d1.dot(d2)+1)/(N*sigma[col])+np.identity(N)*1/sigma[col]).dot(dy[:,col].reshape((-1,1)))))
    # print(dx)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

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
    eps = ln_param.get("eps", 1e-5)
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
    m=np.mean(x,axis=1).reshape((-1,1))
    v=np.var(x,axis=1).reshape((-1,1))
    y=(x-m)/np.sqrt(v+eps)
    out=gamma*y+beta
    cache=(x,gamma,beta,y,eps)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

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

    x,gamma,beta,y,eps=cache
    dbeta=np.sum(dout,axis=0)
    dgamma=np.sum(dout*y,axis=0)
    x=x.transpose()
    y=y.transpose()
    N=x.shape[0]
    D=x.shape[1]
    m=np.mean(x,axis=0)
    v=np.var(x,axis=0)
    sigma=np.sqrt(v+eps)
    dy=gamma*dout
    dy=dy.transpose()
    dx=(-1/N*(np.ones(x.shape)*np.sum(dy,axis=0)+y*np.sum(y*dy,axis=0))+dy)/sigma
    dx=dx.transpose()
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

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
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        mask=(np.random.rand(*x.shape)<p)/p
        out=x*mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out=x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        p= dropout_param["p"]
        dx=mask*dout

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

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
    N,C,H,W=x.shape
    F,_,HH,WW=w.shape

    stride=conv_param['stride']
    pad=conv_param['pad']
    H1=(int)(1+(H+2*pad-HH)/stride)
    W1=(int)(1+(W+2*pad-WW)/stride)
    x_=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
    # print(x)
    for i in range(N):
      f_temp=None
      for f in range(F):
        temp_data=None
        for row in range(H1):
          temp=np.array([])
          for col in range(W1):
            # print(temp)
            temp=np.append(temp,np.sum(x_[i,:,row*stride:row*stride+HH,col*stride:col*stride+WW]*w[f])+b[f])
          # temp=temp.reshape((1,-1))
          temp=temp[np.newaxis,:]
          if temp_data is None:
            temp_data=temp
          else:
            temp_data=np.append(temp_data,temp,axis=0)
          # print(temp_data.shape)
        temp_data=temp_data[np.newaxis,:]
        if f_temp is None:
          f_temp=temp_data
        else:
          f_temp=np.append(f_temp,temp_data,axis=0)
        # print(f_temp.shape)
      # f_temp=f_temp+b
      f_temp=f_temp[np.newaxis,:]
      if out is None:
        out=f_temp
      else:
        out=np.append(out,f_temp,axis=0)
    # out+=b
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

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

    x,w,b,conv_param=cache
    N,C,H,W=x.shape
    F,_,HH,WW=w.shape
    stride=conv_param['stride']
    pad=conv_param['pad']
    H1=(int)(1+(H+2*pad-HH)/stride)
    W1=(int)(1+(W+2*pad-WW)/stride)
    db=np.zeros(b.shape)
    dx=np.zeros(x.shape)
    dw=np.zeros(w.shape)
    db=np.sum(np.sum(np.sum(dout,axis=0),axis=1),axis=1)
    x_=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
    dx_=np.zeros(x_.shape)
    for i in range(N):
      for f in range(F):
        for row in range(H1):
          for col in range(W1):
            dw[f]+=x_[i,:,row*stride:row*stride+HH,col*stride:col*stride+WW]*dout[i,f,row,col]
            dx_[i,:,row*stride:row*stride+HH,col*stride:col*stride+WW]+=w[f]*dout[i,f,row,col]
    dx=np.delete(dx_,[0,-1],axis=-1)
    dx=np.delete(dx,[0,-1],axis=2)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

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

    N,C,H,W=x.shape
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']
    H1=(int)(1+(H-pool_height)/stride)
    W1=(int)(1+(W-pool_width)/stride)
    for i in range(N):
      f_temp=None
      for c in range(C):
        temp_data=None
        for row in range(H1):
          temp=np.array([])
          for col in range(W1):
            temp=np.append(temp,np.max(x[i,c,row*stride:row*stride+pool_height,col*stride:col*stride+pool_width]))
          temp=temp[np.newaxis,:]
          if temp_data is None:
            temp_data=temp
          else:
            temp_data=np.append(temp_data,temp,axis=0)
        temp_data=temp_data[np.newaxis,:]
        if f_temp is None:
          f_temp=temp_data
        else:
          f_temp=np.append(f_temp,temp_data,axis=0)
      f_temp=f_temp[np.newaxis,:]
      if out is None:
        out=f_temp
      else:
        out=np.append(out,f_temp,axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

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
    x,pool_param=cache
    N,C,H,W=x.shape
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']
    H1=(int)(1+(H-pool_height)/stride)
    W1=(int)(1+(W-pool_width)/stride)
    dx=np.zeros(x.shape)
    out=None
    for i in range(N):
      f_temp=None
      for c in range(C):
        temp_data=None
        for row in range(H1):
          temp=np.array([])
          for col in range(W1):
            index=np.argmax(x[i,c,row*stride:row*stride+pool_height,col*stride:col*stride+pool_width])
            index_row=(int)(index/pool_width)
            index_col=index%pool_width
            dx[i,c,row*stride+index_row,col*stride+index_col]=dout[i,c,row,col]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

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
    N,C,H,W=x.shape
    x_=np.swapaxes(x,1,3)
    x_=x_.reshape((-1,C))
    out,cache=batchnorm_forward(x_,gamma,beta,bn_param)
    out=out.reshape((N,W,H,C))
    out=np.swapaxes(out,1,3)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

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
    N,C,H,W=dout.shape
    dout=np.swapaxes(dout,1,3)
    dout=dout.reshape((-1,C))
    dx,dgamma,dbeta=batchnorm_backward_alt(dout,cache)
    dx=dx.reshape((N,W,H,C))
    dx=np.swapaxes(dx,1,3)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape    # x shape: (N, C, H, W)
    x_ = x.reshape((N*G, -1))  # x_ shape: (N*G, C*H*W/G)
                           # Example: if N=2, C=6, H=4, W=5, G=2
                           # Then shape is: (4, 60)

    # Compute mean for each group
    m = np.mean(x_, axis=1).reshape((-1, 1))  # m shape: (N*G, 1)
                                          # Example: (4, 1)

    # Compute variance for each group
    v = np.var(x_, axis=1).reshape((-1, 1))   # v shape: (N*G, 1)
                                          # Example: (4, 1)

    # Normalize each group
    y_ = (x_ - m)/np.sqrt(v + eps)  # y_ shape: same as x_: (N*G, C*H*W/G)
                                # Example: (4, 60)

    # Reshape back to original shape
    y = y_.reshape(x.shape)  # y shape: same as x: (N, C, H, W)
                        # Example: (2, 6, 4, 5)

    # Scale and shift
    out = gamma*y + beta    # out shape: same as x: (N, C, H, W)
                       # gamma and beta broadcast from shape (C,)
                       # Example: (2, 6, 4, 5)

    # Cache for backward pass
    cache = (x_, gamma, beta, y, y_, eps, G)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # N, C, H, W=x.shape
    N, C, H, W=dout.shape
    x_,gamma,beta,y,y_,eps,G=cache
    dbeta=np.sum(np.sum(np.sum(dout,axis=0),axis=1),axis=1).reshape((1,C,1,1))
    dgamma=np.sum(np.sum(np.sum(dout*y,axis=0),axis=1),axis=1).reshape((1,C,1,1))
    x=x_.transpose()
    y=y_.transpose()
    m=np.mean(x,axis=0)
    v=np.var(x,axis=0)
    sigma=np.sqrt(v+eps)
    dy=gamma*dout
    dy=dy.reshape((N*G,-1))
    dy=dy.transpose()
    N=x.shape[0]
    dx=(-1/N*(np.ones(x.shape)*np.sum(dy,axis=0)+y*np.sum(y*dy,axis=0))+dy)/sigma
    dx=dx.transpose()
    dx=dx.reshape(dout.shape)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
