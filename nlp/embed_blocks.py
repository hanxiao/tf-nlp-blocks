import numpy as np
import tensorflow as tf


def Positional_embed(batch_size,
                     seq_length,
                     max_seq_len,
                     num_units,
                     zero_pad=True,
                     scale=True,
                     scope='Pos_Block',
                     reuse=None):
    """Embeds a given tensor.
    Args:
      seqs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      max_seq_len: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
    """
    with tf.variable_scope(scope, reuse=reuse):
        seqs = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])

        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[max_seq_len, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, seqs)

        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs


def SinusPositional_embed(batch_size,
                          seq_length,
                          num_units,
                          zero_pad=True,
                          scale=True,
                          scope='Sinus_Pos_Block',
                          reuse=None):
    """Sinusoidal Positional Encoding.
    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    """

    with tf.variable_scope(scope, reuse=reuse):
        seq = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
            for pos in range(seq_length)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, seq)

        if scale:
            outputs = outputs * num_units ** 0.5

        return outputs
