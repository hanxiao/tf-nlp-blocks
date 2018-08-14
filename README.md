# tf-nlp-blocks
[![Python: >=3.6](https://img.shields.io/badge/Python->=3.6-brightgreen.svg)](https://opensource.org/licenses/MIT)    [![Tensorflow: >=1.6](https://img.shields.io/badge/Tensorflow->=1.6-brightgreen.svg)](https://opensource.org/licenses/MIT)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Han Xiao https://hanxiao.github.io




A collection of frequently-used NLP blocks I have implemented in Tensorflow. All implementations follow a modularize design pattern which I call "block", Details can be found [in my blog post](https://hanxiao.github.io/2018/06/25/4-Encoding-Blocks-You-Need-to-Know-Besides-LSTM-RNN-in-Tensorflow/).

## Requirements

- Python >= 3.6
- Tensorflow >= 1.6

## Content

### `encode_blocks.py`
sequence encoding blocks. Input `[B, L, D]`, output is `[B, L, D']`, where `B` is batch size, `L` is the length of the sequence and `D` and `D'` are the dimensions.

| Name  | Description | Reference |
| --- | --- |--- |
| `LSTM_encode`| a fast multi-layer bidirectional LSTM implementation based on [`CudnnLSTM`](https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnLSTM#call) | [Tensorflow doc on `CudnnLSTM`](https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnLSTM#call)|
| `TCN_encode` | a temporal convolution netowork, basically a multi-layer dilated CNN with special padding to ensure the causality| [Temporal Convolutional Networks: A Unified Approach to Action Segmentation](https://arxiv.org/abs/1608.08242)|
| `Res_DualCNN_encode` | an internal block used in `TCN_encode`. A two-layer CNN with spatial dropout inbetween, followed by a residual connection and a layer-norm.| [Temporal Convolutional Networks: A Unified Approach to Action Segmentation](https://arxiv.org/abs/1608.08242)|
| `CNN_encode` | a standard `conv1d` implementation on `L` axis, with the possibility to set different paddings | [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)|

### `match_blocks.py`


- `embed_blocks.py`: positional encoding on the sequence
    - `Positional_embed`
    - `SinusPositional_embed`: sinusoid encoding described in ["Attention is all you need"](https://arxiv.org/pdf/1706.03762.pdf)
    