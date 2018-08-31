---
layout: post
title: '(작성중)[Tensorflow] Basic tutorial'
categories:
  - Coding
tags:
  - deep learning
  - tensorflow
  - implementation
---


Reference : [Tensorflow Guide](https://www.tensorflow.org/guide/low_level_intro)<br>
`Tensorflow`를 처음 사용하시는 분들을 위한 가이드입니다. 본 포스트에서는 `keras`와 같은 high-level API가 아닌 **low-level API**로 작성합니다.<br>

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
```

