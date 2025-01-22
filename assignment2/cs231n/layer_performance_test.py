import time
import numpy as np
from cs231n.layers import conv_forward_naive, conv_backward_naive
from cs231n.fast_layers import conv_forward_fast, conv_backward_fast
from cs231n.layers import max_pool_forward_naive, max_pool_backward_naive
from cs231n.fast_layers import max_pool_forward_fast, max_pool_backward_fast

def rel_error(x, y):
    """ Returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def time_function(f, *args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
    """
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

def compare_conv_implementations():
    np.random.seed(231)
    
    # Test conv_forward_fast
    x = np.random.randn(100, 3, 31, 31)
    w = np.random.randn(25, 3, 3, 3)
    b = np.random.randn(25,)
    conv_param = {'stride': 2, 'pad': 1}
    
    print('\nTesting conv_forward_fast:')
    print('Naive: %fs' % time_function(conv_forward_naive, x, w, b, conv_param))
    print('Fast: %fs' % time_function(conv_forward_fast, x, w, b, conv_param))
    
    out_naive, cache_naive = conv_forward_naive(x, w, b, conv_param)
    out_fast, cache_fast = conv_forward_fast(x, w, b, conv_param)
    print('Difference: ', rel_error(out_naive, out_fast))
    
    print('\nTesting conv_backward_fast:')
    dout = np.random.randn(*out_naive.shape)
    print('Naive: %fs' % time_function(conv_backward_naive, dout, cache_naive))
    print('Fast: %fs' % time_function(conv_backward_fast, dout, cache_fast))
    
    dx_naive, dw_naive, db_naive = conv_backward_naive(dout, cache_naive)
    dx_fast, dw_fast, db_fast = conv_backward_fast(dout, cache_fast)
    print('dx difference: ', rel_error(dx_naive, dx_fast))
    print('dw difference: ', rel_error(dw_naive, dw_fast))
    print('db difference: ', rel_error(db_naive, db_fast))

def compare_pool_implementations():
    np.random.seed(231)
    
    x = np.random.randn(100, 3, 32, 32)
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    print('\nTesting max_pool_forward_fast:')
    print('Naive: %fs' % time_function(max_pool_forward_naive, x, pool_param))
    print('Fast: %fs' % time_function(max_pool_forward_fast, x, pool_param))
    
    out_naive, cache_naive = max_pool_forward_naive(x, pool_param)
    out_fast, cache_fast = max_pool_forward_fast(x, pool_param)
    print('Difference: ', rel_error(out_naive, out_fast))
    
    print('\nTesting max_pool_backward_fast:')
    dout = np.random.randn(*out_naive.shape)
    print('Naive: %fs' % time_function(max_pool_backward_naive, dout, cache_naive))
    print('Fast: %fs' % time_function(max_pool_backward_fast, dout, cache_fast))
    
    dx_naive = max_pool_backward_naive(dout, cache_naive)
    dx_fast = max_pool_backward_fast(dout, cache_fast)
    print('dx difference: ', rel_error(dx_naive, dx_fast))

if __name__ == '__main__':
    print("Comparing Convolution Implementations:")
    compare_conv_implementations()
    
    print("\nComparing Pooling Implementations:")
    compare_pool_implementations() 