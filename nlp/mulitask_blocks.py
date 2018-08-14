#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>


import tensorflow as tf

from nlp.nn import get_var, linear_logit, get_cross_correlated_mat, get_self_correlated_mat, gate_filter


def CrossStitch(logit_A, logit_B,
                pA_given_B, pB_given_A,
                pA_corr=None, pB_corr=None,
                reduce_transfer='LEARNED',
                scope=None, reuse=None,
                **kwargs):
    with tf.variable_scope(scope or 'Cross_Stitch_Block', reuse=reuse):
        trans_B = tf.matmul(logit_A, pB_given_A)  # batch x num B
        trans_A = tf.matmul(logit_B, pA_given_B, transpose_b=True)  # batch x num A

        if pA_corr is not None:
            logit_A = tf.matmul(logit_A, pA_corr)
        if pB_corr is not None:
            logit_B = tf.matmul(logit_B, pB_corr)

        if reduce_transfer == 'LEARNED':
            res_rate_A = tf.sigmoid(get_var('taskA_weight', shape=[1, logit_A.get_shape().as_list()[-1]]))
            res_rate_B = tf.sigmoid(get_var('taskB_weight', shape=[1, logit_B.get_shape().as_list()[-1]]))
            output_A = res_rate_A * logit_A + (1 - res_rate_A) * trans_A  # batch x num A
            output_B = res_rate_B * logit_B + (1 - res_rate_B) * trans_B  # batch x num B
        elif reduce_transfer == 'MEAN':
            output_A = (logit_A + trans_A) / 2
            output_B = (logit_B + trans_B) / 2
        elif reduce_transfer == 'GATE':
            output_A = logit_A * tf.sigmoid(trans_A)
            output_B = logit_B * tf.sigmoid(trans_B)
        elif reduce_transfer == 'RES_GATE':
            output_A = logit_A + logit_A * tf.sigmoid(trans_A)
            output_B = logit_B + logit_B * tf.sigmoid(trans_B)
        else:
            raise NotImplementedError

        return output_A, output_B


def Stack_CrossStitch_old(shared_input, cooc_AB=None, num_layers=3,
                          num_out_A=None, num_out_B=None,
                          transform=linear_logit,
                          learn_cooc='FIXED',
                          gated=False,
                          self_correlation=False,
                          scope=None,
                          reuse=None,
                          **kwargs):
    with tf.variable_scope(scope or 'Stacked_Cross_Stitch_Block', reuse=reuse):
        pA_given_B, pB_given_A = get_cross_correlated_mat(num_out_A, num_out_B, learn_cooc, cooc_AB)

        pA_corr = get_self_correlated_mat(num_out_A, scope='self_corr_a') if self_correlation else None
        pB_corr = get_self_correlated_mat(num_out_B, scope='self_corr_b') if self_correlation else None

        iter_shared = shared_input
        out_A, out_B = None, None
        for j in range(num_layers):
            with tf.variable_scope('cross_stitch_block_%d' % j):
                iter_inputA = gate_filter(iter_shared, 'threshold_a') if gated else iter_shared
                iter_inputB = gate_filter(iter_shared, 'threshold_b') if gated else iter_shared
                logit_A = transform(iter_inputA, num_out_A, scope='dense_A', **kwargs)
                logit_B = transform(iter_inputB, num_out_B, scope='dense_B', **kwargs)
                out_A, out_B = CrossStitch(logit_A, logit_B,
                                           pA_given_B, pB_given_A,
                                           pA_corr, pB_corr,
                                           **kwargs)
                iter_shared = tf.concat([out_A, out_B], axis=-1)
        return out_A, out_B


def Stack_CrossStitch(shared_input=None, init_input_A=None, init_input_B=None,
                      cooc_AB=None, num_layers=3,
                      num_out_A=None, num_out_B=None,
                      transform=linear_logit,
                      learn_cooc='FIXED',
                      self_correlation=False,
                      scope=None,
                      reuse=None,
                      **kwargs):
    if shared_input is None and (init_input_A is None or init_input_B is None):
        raise AttributeError('Must specify shared input or init_input')

    with tf.variable_scope(scope or 'Stacked_Cross_Stitch_Block', reuse=reuse):
        pA_given_B, pB_given_A = get_cross_correlated_mat(num_out_A, num_out_B, learn_cooc, cooc_AB)

        pA_corr = get_self_correlated_mat(num_out_A, scope='self_corr_a') if self_correlation else None
        pB_corr = get_self_correlated_mat(num_out_B, scope='self_corr_b') if self_correlation else None

        iter_inputA = init_input_A if init_input_A is not None else shared_input
        iter_inputB = init_input_B if init_input_B is not None else shared_input

        out_A, out_B = None, None
        for j in range(num_layers):
            with tf.variable_scope('cross_stitch_block_%d' % j):
                logit_A = transform(iter_inputA, num_out_A, scope='dense_A', **kwargs)
                logit_B = transform(iter_inputB, num_out_B, scope='dense_B', **kwargs)
                out_A, out_B = CrossStitch(logit_A, logit_B,
                                           pA_given_B, pB_given_A,
                                           pA_corr, pB_corr,
                                           **kwargs)
                iter_inputA = tf.concat([out_A, out_B], axis=-1)
                iter_inputB = iter_inputA
        return out_A, out_B
