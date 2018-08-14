#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

import numpy as np
import tensorflow as tf

from nlp.match_blocks import AttentiveCNN_match

if __name__ == '__main__':
    # build two batch sequences
    # context batch has 2 seqs, with length 4 and 3, padded to 5
    # query batch has 2 seqs, with length 2 and 3, padded to 3
    # dimension of each symbol is 4
    context = tf.constant(np.random.random([2, 5, 4]), dtype=tf.float32)  # B,Lc,D
    query = tf.constant(np.random.random([2, 3, 4]), dtype=tf.float32)  # B,Lq,D
    # build mask
    context_mask = tf.sequence_mask([4, 3], 5, dtype=tf.float32, name='query_mask')
    query_mask = tf.sequence_mask([2, 3], 3, dtype=tf.float32, name='query_mask')

    # standard att-cnn
    output1 = AttentiveCNN_match(context, query, context_mask, query_mask, scope='attcnn1')
    # with res and layernorm
    output2 = AttentiveCNN_match(context, query, context_mask, query_mask,
                                 residual=True, normalize_output=True, scope='attcnn2')
    # ensure causality
    output3 = AttentiveCNN_match(context, query, context_mask, query_mask,
                                 casuality=True, scope='attcnn3')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for v in sess.run([output1, output2, output3]):
            print(v)
