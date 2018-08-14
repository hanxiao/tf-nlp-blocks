# tf-nlp-blocks
[![Python: >=3.6](https://img.shields.io/badge/Python->=3.6-brightgreen.svg)](https://opensource.org/licenses/MIT)    [![Tensorflow: >=1.6](https://img.shields.io/badge/Tensorflow->=1.6-brightgreen.svg)](https://opensource.org/licenses/MIT)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Author: Han Xiao https://hanxiao.github.io


A collection of frequently-used deep learning blocks I have implemented in Tensorflow. It covers the core tasks in NLP such as embedding, encoding, matching and pooling. All implementations follow a modularized design pattern which I called "block-design". More details can be found [in my blog post](https://hanxiao.github.io/2018/06/25/4-Encoding-Blocks-You-Need-to-Know-Besides-LSTM-RNN-in-Tensorflow/).

## Requirements

- Python >= 3.6
- Tensorflow >= 1.6

## Contents

### `encode_blocks.py`
A collection of sequence encoding blocks. Input is a sequence with shape of `[B, L, D]`, output is another sequence in `[B, L, D']`, where `B` is batch size, `L` is the length of the sequence and `D` and `D'` are the dimensions.

| Name  | Dependencies| Description | Reference |
| --- | --- |--- |--- |
| `LSTM_encode`| | a fast multi-layer bidirectional LSTM implementation based on [`CudnnLSTM`](https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnLSTM#call) | [Tensorflow doc on `CudnnLSTM`](https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnLSTM#call)|
| `TCN_encode` | `Res_DualCNN_encode`| a temporal convolution netowork, basically a multi-layer dilated CNN with special padding to ensure the causality| [Temporal Convolutional Networks: A Unified Approach to Action Segmentation](https://arxiv.org/abs/1608.08242)|
| `Res_DualCNN_encode` |`CNN_encode`| a sub-block used by `TCN_encode`. It is a two-layer CNN with spatial dropout in-between, then followed by a residual connection and a layer-norm.| [Temporal Convolutional Networks: A Unified Approach to Action Segmentation](https://arxiv.org/abs/1608.08242)|
| `CNN_encode` | | a standard `conv1d` implementation on `L` axis, with the possibility to set different paddings | [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)|

### `match_blocks.py`
A collection of sequence matching blocks, aka. attention. Input are two sequnces: `context` in the shape of `[B, L_c, D]`, and `query` in the shape of `[B, L_q, D]`. The output is a sequence has the same length as `context`, i.e. with shape of `[B, L_c, D]`. Each position in the output should encodes the relevance of that position in `context` to the complete `query`.

| Name  | Dependencies | Description | Reference |
| --- | --- |--- |--- |
|`Attentive_match`| |basic attention mechanism with different scoring functions, also supports future blinding.| `additive`: [Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473); `scaled`: [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)| 
|`Transformer_match`| |a multi-head attention block from ["Attention is all you need"](https://arxiv.org/pdf/1706.03762.pdf)| [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)|
|`AttentiveCNN_match`| `Attentive_match`|the light version of attentive convolution, with the possibility of future blinding to ensure causality. | [Attentive Convolution](https://arxiv.org/pdf/1710.00519)
|`BiDaf_match`| |attention flow layer used in bidaf model. | [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603)|

### `pool_blocks.py`
A collection of pooling blocks. It fuses/reduces on the time axis `L`. Input is a sequence with shape of `[B, L, D]`, output is in `[B, D]`.

| Name  | Dependencies | Description | Reference |
| --- | --- |--- |--- |
|`SWEM_pool`| | do pooling on the input sequence, supports max/avg. pooling, hierarchical avg. max pooling. | [Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms](https://arxiv.org/abs/1805.09843) |

There are also some convolution-based pooling blocks build on `SWEM_pool`, but they are for experimental purpose. Thus, I will not list them here.

### `embed_blocks.py`
A collection of positional encoding on the sequence.

| Name  | Dependencies | Description | Reference |
| --- | --- |--- |--- |
|`SinusPositional_embed`| | generate a sinusoid signal that has the same length of the input sequence | [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)|
|`Positional_embed`| |parameterize the absolute position of the tokens in the input sequence | [A Convolutional Encoder Model for Neural Machine Translation](https://arxiv.org/pdf/1611.02344.pdf)|

### `mulitask_blocks.py`
A collection of multi-task learning blocks

| Name  | Dependencies | Description | Reference |
| --- | --- |--- |--- |
|`CrossStitch`||a cross-stitch block, modeling the correlation & self-correlation of two tasks| [Cross-stitch Networks for Multi-task Learning](https://arxiv.org/pdf/1604.03539)|
|`Stack_CrossStitch`|`CrossStitch`|stacking multiple cross-stitch blocks together with shared/separated input| [Cross-stitch Networks for Multi-task Learning](https://arxiv.org/pdf/1604.03539)|


### `nn.py`
A collection of auxiliary functions, e.g. masking, normalizing, slicing. 


## Run 

Run `app.py` for a simple test on toy data.