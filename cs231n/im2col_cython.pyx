import numpy as np
cimport numpy as np

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

def im2col_cython(np.ndarray[DTYPE_t, ndim=4] x, int field_height,
                  int field_width, int padding, int stride):
    cdef int N = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]
    
    cdef int HH = (H + 2 * padding - field_height) / stride + 1
    cdef int WW = (W + 2 * padding - field_width) / stride + 1

    cdef int p = padding
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.pad(x,
            ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    cdef np.ndarray[DTYPE_t, ndim=2] cols = np.zeros(
            (C * field_height * field_width, N * HH * WW),
            dtype=x.dtype)

    # Moving im2col forward pass to Python
    cols = im2col_6d_cython(x_padded, N, C, H, W, HH, WW,
                            field_height, field_width, padding, stride)
    return cols

def col2im_cython(np.ndarray[DTYPE_t, ndim=2] cols, int N, int C, int H, int W,
                  int field_height, int field_width, int padding, int stride):
    cdef int HH = (H + 2 * padding - field_height) / stride + 1
    cdef int WW = (W + 2 * padding - field_width) / stride + 1
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding),
                                        dtype=cols.dtype)
    cdef int c, ii, jj, row, yy, xx, i, j
    
    for c in range(C):
        for ii in range(field_height):
            for jj in range(field_width):
                row = c * field_height * field_width + ii * field_width + jj
                for yy in range(HH):
                    for xx in range(WW):
                        for i in range(N):
                            col_idx = i * HH * WW + yy * WW + xx
                            ii_padded = stride * yy + ii
                            jj_padded = stride * xx + jj
                            x_padded[i, c, ii_padded, jj_padded] += cols[row, col_idx]

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded

def im2col_6d_cython(np.ndarray[DTYPE_t, ndim=4] x_padded,
                      int N, int C, int H, int W, int HH, int WW,
                      int field_height, int field_width, int padding, int stride):
    cdef np.ndarray[DTYPE_t, ndim=2] cols = np.zeros(
            (C * field_height * field_width, N * HH * WW),
            dtype=x_padded.dtype)
    
    cdef int c, ii, jj, row, yy, xx, i, j
    
    for c in range(C):
        for ii in range(field_height):
            for jj in range(field_width):
                row = c * field_height * field_width + ii * field_width + jj
                for yy in range(HH):
                    for xx in range(WW):
                        for i in range(N):
                            col_idx = i * HH * WW + yy * WW + xx
                            ii_padded = stride * yy + ii
                            jj_padded = stride * xx + jj
                            cols[row, col_idx] = x_padded[i, c, ii_padded, jj_padded]
    
    return cols 