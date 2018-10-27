#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>


import tensorflow as tf

from nlp.nn import initializer, regularizer, spatial_dropout, get_lstm_init_state, dropout_res_layernorm


def LSTM_encode(seqs, causality=False, scope='lstm_encode_block', reuse=None, **kwargs):
    with tf.variable_scope(scope, reuse=reuse):
        if causality:
            kwargs['direction'] = 'unidirectional'
        if 'num_units' not in kwargs or kwargs['num_units'] is None:
            kwargs['num_units'] = seqs.get_shape().as_list()[-1]
        batch_size = tf.shape(seqs)[0]
        _seqs = tf.transpose(seqs, [1, 0, 2])  # to T, B, D
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(**kwargs)
        init_state = get_lstm_init_state(batch_size, **kwargs)
        output = lstm(_seqs, init_state)[0]  # 2nd return is state, ignore
        return tf.transpose(output, [1, 0, 2])  # back to B, T, D


def TCN_encode(seqs, num_layers, scope='tcn_encode_block', reuse=None, **kwargs):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = [seqs]
        for i in range(num_layers):
            dilation_size = 2 ** i
            out = Res_DualCNN_encode(outputs[-1], dilation=dilation_size, scope='res_biconv_%d' % i, **kwargs)
            outputs.append(out)
        return outputs[-1]


def Res_DualCNN_encode(seqs, use_spatial_dropout=True, scope='res_biconv_block', reuse=None, **kwargs):
    with tf.variable_scope(scope, reuse=reuse):
        out1 = CNN_encode(seqs, scope='first_conv1d', **kwargs)
        if use_spatial_dropout:
            out1 = spatial_dropout(out1)
        out2 = CNN_encode(out1, scope='second_conv1d', **kwargs)
        if use_spatial_dropout:
            out2 = CNN_encode(out2)
        return dropout_res_layernorm(seqs, out2, **kwargs)


def CNN_encode(seqs, filter_size=3, dilation=1,
               num_filters=None, direction='forward',
               causality=False,
               act_fn=tf.nn.relu,
               scope=None,
               reuse=None, **kwargs):
    input_dim = seqs.get_shape().as_list()[-1]
    num_filters = num_filters if num_filters else input_dim

    # add causality: shift the whole seq to the right
    if causality:
        direction = 'forward'
    padding = (filter_size - 1) * dilation
    if direction == 'forward':
        pad_seqs = tf.pad(seqs, [[0, 0], [padding, 0], [0, 0]])
        padding_scheme = 'VALID'
    elif direction == 'backward':
        pad_seqs = tf.pad(seqs, [[0, 0], [0, padding], [0, 0]])
        padding_scheme = 'VALID'
    elif direction == 'none':
        pad_seqs = seqs  # no padding, must set to SAME so that we have same length
        padding_scheme = 'SAME'
    else:
        raise NotImplementedError

    with tf.variable_scope(scope or 'causal_conv_%s_%s' % (filter_size, direction), reuse=reuse):
        return tf.layers.conv1d(
            pad_seqs,
            num_filters,
            filter_size,
            activation=act_fn,
            padding=padding_scheme,
            dilation_rate=dilation,
            kernel_initializer=initializer,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=regularizer)
