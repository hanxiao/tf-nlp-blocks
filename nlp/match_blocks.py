#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>


import tensorflow as tf

from nlp.encode_blocks import CNN_encode
from nlp.nn import linear_logit, layer_norm, dropout_res_layernorm


def AttentiveCNN_match(context, query, context_mask,
                       query_mask, causality=False,
                       scope='AttentiveCNN_Block',
                       reuse=None, **kwargs):
    with tf.variable_scope(scope, reuse=reuse):
        direction = 'forward' if causality else 'none'
        cnn_wo_att = CNN_encode(context, filter_size=3, direction=direction, act_fn=None)
        att_context, _ = Attentive_match(context, query, context_mask, query_mask, causality=causality)
        cnn_att = CNN_encode(att_context, filter_size=1, direction=direction, act_fn=None)
        output = tf.nn.tanh(cnn_wo_att + cnn_att)
        return dropout_res_layernorm(context, output, **kwargs)


def Attentive_match(context, query, context_mask, query_mask,
                    num_units=None,
                    score_func='scaled', causality=False,
                    scope='attention_match_block', reuse=None, **kwargs):
    with tf.variable_scope(scope, reuse=reuse):
        batch_size, context_length, _ = context.get_shape().as_list()
        if num_units is None:
            num_units = context.get_shape().as_list()[-1]
        _, query_length, _ = query.get_shape().as_list()

        context = linear_logit(context, num_units, act_fn=tf.nn.relu, scope='context_mapping')
        query = linear_logit(query, num_units, act_fn=tf.nn.relu, scope='query_mapping')

        if score_func == 'dot':
            score = tf.matmul(context, query, transpose_b=True)
        elif score_func == 'bilinear':
            score = tf.matmul(linear_logit(context, num_units, scope='context_x_We'), query, transpose_b=True)
        elif score_func == 'scaled':
            score = tf.matmul(linear_logit(context, num_units, scope='context_x_We'), query, transpose_b=True) / \
                    (num_units ** 0.5)
        elif score_func == 'additive':
            score = tf.squeeze(linear_logit(
                tf.tanh(tf.tile(tf.expand_dims(linear_logit(context, num_units, scope='context_x_We'), axis=2),
                                [1, 1, query_length, 1]) +
                        tf.tile(tf.expand_dims(linear_logit(query, num_units, scope='query_x_We'), axis=1),
                                [1, context_length, 1, 1])), 1, scope='x_ve'), axis=3)
        else:
            raise NotImplementedError

        mask = tf.matmul(tf.expand_dims(context_mask, -1), tf.expand_dims(query_mask, -1), transpose_b=True)
        paddings = tf.ones_like(mask) * (-2 ** 32 + 1)
        masked_score = tf.where(tf.equal(mask, 0), paddings, score)  # B, Lc, Lq

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(masked_score[0, :, :])  # (Lc, Lq)
            tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (Lc, Lq)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(masked_score)[0], 1, 1])  # B, Lc, Lq

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            masked_score = tf.where(tf.equal(masks, 0), paddings, masked_score)  # B, Lc, Lq

        query2context_score = tf.nn.softmax(masked_score, axis=2) * mask  # B, Lc, Lq
        query2context_attention = tf.matmul(query2context_score, query)  # B, Lc, D

        context2query_score = tf.nn.softmax(masked_score, axis=1) * mask  # B, Lc, Lq
        context2query_attention = tf.matmul(context2query_score, context, transpose_a=True)  # B, Lq, D

        return (query2context_attention,  # B, Lc, D
                context2query_attention)  # B, Lq, D


def Transformer_match(context,
                      query,
                      context_mask,
                      query_mask,
                      num_units=None,
                      num_heads=4,
                      dropout_keep_rate=1.0,
                      causality=False,
                      scope='Transformer_Block',
                      reuse=None,
                      residual=False,
                      normalize_output=False,
                      **kwargs):
    """Applies multihead attention.

    Args:
      context: A 3d tensor with shape of [N, T_q, C_q].
      query: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    """
    if num_units is None or residual:
        num_units = context.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units

        # Linear projections
        Q = tf.layers.dense(context, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(query, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(query, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking, aka query
        if query_mask is None:
            query_mask = tf.sign(tf.abs(tf.reduce_sum(query, axis=-1)))  # (N, T_k)

        mask1 = tf.tile(query_mask, [num_heads, 1])  # (h*N, T_k)
        mask1 = tf.tile(tf.expand_dims(mask1, 1), [1, tf.shape(context)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(mask1, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking  aka, context
        if context_mask is None:
            context_mask = tf.sign(tf.abs(tf.reduce_sum(context, axis=-1)))  # (N, T_q)

        mask2 = tf.tile(context_mask, [num_heads, 1])  # (h*N, T_q)
        mask2 = tf.tile(tf.expand_dims(mask2, -1), [1, 1, tf.shape(query)[1]])  # (h*N, T_q, T_k)
        outputs *= mask2  # (h*N, T_q, T_k)

        # Dropouts
        outputs = tf.nn.dropout(outputs, keep_prob=dropout_keep_rate)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        if residual:
            # Residual connection
            outputs += context

        if normalize_output:
            # Normalize
            outputs = layer_norm(outputs)  # (N, T_q, C)

    return outputs


def BiDaf_match(context, query, context_mask, query_mask, scope=None, reuse=None, **kwargs):
    # context: [batch, l, d]
    # question: [batch, l2, d]
    with tf.variable_scope(scope, reuse=reuse):
        n_a = tf.tile(tf.expand_dims(context, 2), [1, 1, tf.shape(query)[1], 1])
        n_b = tf.tile(tf.expand_dims(query, 1), [1, tf.shape(context)[1], 1, 1])

        n_ab = n_a * n_b
        n_abab = tf.concat([n_ab, n_a, n_b], -1)

        kernel = tf.squeeze(tf.layers.dense(n_abab, units=1), -1)

        context_mask = tf.expand_dims(context_mask, -1)
        query_mask = tf.expand_dims(query_mask, -1)
        kernel_mask = tf.matmul(context_mask, query_mask, transpose_b=True)
        kernel += (kernel_mask - 1) * 1e5

        con_query = tf.matmul(tf.nn.softmax(kernel, 1), query)
        con_query = con_query * context_mask

        query_con = tf.matmul(tf.transpose(
            tf.reduce_max(kernel, 2, keepdims=True), [0, 2, 1]), context * context_mask)
        query_con = tf.tile(query_con, [1, tf.shape(context)[1], 1])
        return tf.concat([context * query_con, context * con_query, context, query_con], 2)


def Stacked_Self_match(context, context_mask, self_match,
                       num_layers=3,
                       scope='Stacked_Self_match_Block',
                       reuse=None, **kwargs):
    output = context
    with tf.variable_scope(scope, reuse=reuse):
        for j in range(num_layers):
            output = self_match(output, output, context_mask, context_mask,
                                scope=scope + '_layer_%d' % j, **kwargs)
    return output
