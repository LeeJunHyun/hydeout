---
layout: post
title: '(작성중)[Tensorflow] Fashion-MNIST tutorial'
categories:
  - Coding
tags:
  - deep learning
  - tensorflow
  - implementation
---

Reference : [Tensorflow Basic classification](https://www.tensorflow.org/tutorials/keras/basic_classification)<br>
`Tensorflow`를 이용하여 Fashion-MNIST dataset을 classification 하는 튜토리얼입니다.<br>

![Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist/blob/master/doc/img/fashion-mnist-sprite.png?raw=true)

```python
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
```
필요한 라이브러리들을 import 합니다.<br>
`keras`를 사용하면 high-level 모듈을 불러와서 학습모델을 쉽게 만들 수 있습니다.<br>
`numpy`는 행렬연산을 위한 파이썬 라이브러리입니다.<br>
`matplotlib.pyplot`는 시각화를 위한 파이썬 라이브러리입니다.

