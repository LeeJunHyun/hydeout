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

Reference : [Tensorflow Basic Classification](https://www.tensorflow.org/tutorials/keras/basic_classification)<br>
`Tensorflow`를 이용하여 Fashion-MNIST dataset을 classification 해보는 튜토리얼입니다. 좀더 기초적인 low-level API 가이드는 [Tensorflow  Guide](https://leejunhyun.github.io/coding/2018/09/01/TF-introduction.html)를 참조하시기 바랍니다.<br>

![Fashion-MNIST](/assets/img/Tensorflow/Fashion-MNIST.png)
*Fashion-MNIST dataset*
---
```python
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
```
필요한 라이브러리들을 import 한다.<br>
`keras`를 사용하면 high-level 모듈을 불러와서 학습모델을 쉽게 만들 수 있다.<br>
`numpy`는 행렬연산을 위한 파이썬 라이브러리다.<br>
`matplotlib.pyplot`는 시각화를 위한 파이썬 라이브러리다.

