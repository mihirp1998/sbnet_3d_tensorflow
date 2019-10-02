import numpy as np
import tensorflow as tf

sbnet_module = tf.load_op_library('libsbnet.so')

import ipdb
st = ipdb.set_trace
def divup(a, b):
    return (a+b-1) // b

# Specify input tensor dimensions and block-sparsity parameters
batch = 4
hw = 256
channels = 64
blockSize = [16, 16]
blockStride = [14, 14]
blockOffset = [0, 0]
blockCount = [divup(hw, blockStride[0]), divup(hw, blockStride[1])]

# build kwargs to simplify op calls
inBlockParams = { "dynamic_bsize": blockSize, "dynamic_boffset": blockOffset, "dynamic_bstride": blockStride }
outBlockParams = { "dynamic_bsize": [blockSize[0]-2, blockSize[1]-2], "dynamic_boffset": blockOffset, "dynamic_bstride": blockStride }

# create a random mask representing attention/a priori sparsity
# threshold the mask to a specified percentile sparsity
mask = np.random.randn(batch, blockCount[0], blockCount[1], channels).astype(np.float32)
threshold = np.percentile(mask, 90)
sparseMask = np.greater(mask, threshold).astype(np.float32)

# upsample the mask to full resolution
upsampledMask = sparseMask.repeat(blockStride[0], axis=1).repeat(blockStride[1], axis=2)

# create a random input tensor
x = tf.constant( np.random.randn(batch, hw, hw, channels).astype(np.float32) )

# create a random weight tensor
w = tf.constant( np.random.randn(3, 3, channels, channels).astype(np.float32) )
# bsize=bsize, bsize_out=bsize_out, boffset=boffset, bcount=bcount, bstrides=bstrides)

# reduce the mask to indices by using a fused pooling+indexing operation
indices = sbnet_module.reduce_mask(mask, blockCount, tol=0.5,avgpool=False, **inBlockParams)
# st()
# # stack active overlapping tiles to batch dimension
blockStack = sbnet_module.sparse_gather(x, indices.bin_counts,\
 indices.active_block_indices, transpose=True, **inBlockParams)

# # perform dense convolution on a sparse stack of tiles
convBlocks = tf.nn.conv2d(
    blockStack, w, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW')

# # write/scatter the tiles back on top of original tensor
# # note that the output tensor is reduced by 1 on each side due to 'VALID' convolution
validX = x[:, 1:hw-1, 1:hw-1, :]
y = sbnet_module.sparse_scatter(
    convBlocks, indices.bin_counts, indices.active_block_indices,
    validX, transpose=True, add=False, atomic=False, **outBlockParams)

sess = tf.Session()
indices_n, = sess.run([indices])
block_stak, = sess.run([blockStack])
convblocks, = sess.run([convBlocks])
y_output, = sess.run([y])
st()