---
layout: post
title: '(작성중)[TensorFlow] Basic tutorial'
categories:
  - Coding
tags:
  - deep learning
  - tensorflow
  - implementation
---


Reference : [TensorFlow Guide](https://www.tensorflow.org/guide/low_level_intro)<br>
`tensorflow`를 처음 사용하시는 분들을 위한 가이드입니다. 본 포스트에서는 `keras`와 같은 high-level API가 아닌 **low-level API**로 작성합니다.<br>

`tf.Graph` 와 `tf.Session`을 이용하여 학습을 돌립니다.

---

### Setup
```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
```
필요한 모듈들을 불러온다. `__future__`모듈을 가져오는 것은 호환성있는 코드를 작성하기 위해서다. `absolute_import`는 패키지 내부 모듈과 표준 라이브러리가 겹치는 경우를 해결하기 위해 사용한다. `division`은 /연산자로 나누면 실수를 출력하고 //연산자로 나누면 정수를 출력하게 해준다. `print_function`은 print함수에서 괄호안의 인자를 재지정, 출력분리해준다. `numpy`는 행렬연산에 필요한 라이브러리다.

### Tensor Values
```python
3. # a rank 0 tensor; a scalar with shape [],
[1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
```
TensorFlow 데이터의 단위체는 **tensor**이다. tensor은 임의의 차원을 가진 행렬의 형태를 갖고 있다. tensor의 **rank**는 차원의 수를 의미한다. **shape**은 tensor의 각 차원의 길이를 나타내는 정수 튜플이다. Tensorflow는 tensor의 값을 표현하기 위해 numpy array를 사용한다.

### TensorFlow Core Walkthrough
TensorFlow는 다음 두가지 단계로 구성된다.
1. Computation graph를 구축한다. (`tf.Graph`)
2. Computation graph를 돌린다. (`tf.Session`)

#### Graph
Computation graph란, TensorFlow 연산들을 그래프 형태로 정렬한 것이다. graph는 두 종류의 objects로 이루어져있다.
* Operations(ops) : 그래프의 노드에 해당된다. Operations는 tensor를 인풋으로 받아 계산한 결과를 아웃풋으로 반환한다.
* Tensors : 그래프의 엣지에 해당된다. 그래프를 타고 흐르는 값을 표현한다. 대부분의 TensorFlow 함수들은 `tf.Tensors`를 반환한다.

> `tf.Tensors`는 값을 갖지 않는다. 이건 단지 Computation graph에서 요소들을 다루기 위한 것이다.

간단한 Computation graph를 만들어보자.가장 기초적인 operation은 constant이다. 입력값을 그대로 받아 tensor을 만들어준다. 그러나 위에서 언급했듯이, operation은 값을 갖지 않는다. 그저 자리만 만들어둘 뿐이다. 추후에 Session run을 하게 되면, 그때 값을 보낸다.
```python
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)
```
print함수의 결과는 아래와 같다.

```python
Tensor("Const:0", shape=(), dtype=float32)
Tensor("Const_1:0", shape=(), dtype=float32)
Tensor("add:0", shape=(), dtype=float32)
```
`tf.constant`라는 operation은 값을 갖지 않기때문에 `3.0`,`4.0`,`7.0`을 반환하는게 아니라 그저 computation graph의 노드들을 반환한다.

각각의 operation은 각각의 이름을 갖는다. 이 이름은 Python에서 할당되는 객체의 이름과는 무관하다. Tensors는 위에서 출력된 `add:0` 처럼 operation과 index를 이름으로 갖는다. 

#### TensorBoard
TensorFlow는 TensorBoard라는 유틸리티를 제공한다. TensorBoard의 여러 기능중 하나는 computation graph를 보여주는 것이다. 아래처럼 간단하게 구현할 수 있다.

우선 computation graph를 TensorBoard summary file에 저장한다.
```python
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
```
이것은 현재 경로에 다음과같은 포맷의 이름을 가진 `event` file을 생성한다.
```python
events.out.tfevents.{timestamp}.{hostname}
```
새로운 터미널에서 TensorBoard를 실행시킨다.
```python
tensorboard --logdir .
```
TensorBoard의 **graphs page**를 열어보면, 다음과같은 그래프가 생성되었음을 확인할 수 있다.
![computation graph](/assets/img/TensorFlow/TF-introduction-01.png)


#### Session
tensors을 계산하기 위해, `tf.Session` 객체를 초기화한다. session에는 TensorFlow operation을 돌리는 TensorFlow runtime이 내장되어있다. `tf.Graph`를 `*.py`파일이라고 생각한다면, `tf.Session`은 `python` 실행명령이라고 생각하면 된다.

아래 코드는 `tf.Session`을 만들고, `sess.run`을 통해 위에서 정의한 `total` tensor을 계산한다.
```python
sess = tf.Session()
print(sess.run(total))
```
결과값으로 `7.0`이 나오는 것을 확인할 수 있다.
```ptyhon
7.0
```
`tf.Session.run`을 이용해 여러 tensors를 돌릴 수도 있다. 아래 코드처럼 튜플이나 딕셔너리의 조합을 사용한다.
```python
print(sess.run({'ab':(a, b), 'total':total}))
```
아래와 같은 결과를 확인할 수 있다.
```python
{'total': 7.0, 'ab': (3.0, 4.0)}
```

`tf.Session.run`을 호출할 때, `tf.Tensor`은 오직 하나의 값만 갖는다. 예를들면 아래 코드에서, `tf.random_uniform`은 3개의 랜덤([0,1)의 범위)값을 생성하는 `tf.Tensor`을 만들어낸다.

```python
vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))
```
이때 결과는 아래와 같다.
```python
[ 0.52917576  0.64076328  0.68353939]
[ 0.66192627  0.89126778  0.06254101]
(
  array([ 1.88408756,  1.87149239,  1.84057522], dtype=float32),
  array([ 2.88408756,  2.87149239,  2.84057522], dtype=float32)
)
```
`tf.Session.run` 한번 실행할 때마다 `vec`의 값은 한번씩만 랜덤값을 뽑아내기 때문에, `out1`과 `out2`를 계산할 때는 `vec`의 값이 같음을 알 수 있다.


#### Feeding
위에서는 `tf.Constant`를 통해 `3.0` 이나 `4.0` 처럼 고정된 값을 이용해서 연산을 했다. 이번에는 그때그때 다른 입력값(데이터)을 넣어줄 수 있는 `tf.placeholder`에 대해 소개한다. **placeholder**은 지금 당장 값을 설정하는 것이 아니라, 추후에 값을 넣어주기 위해서 자리를 만들어두는 것이다.

```python
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y
```
이때 `x`와 `y`는 나중에 `sess.run`을 할 때 `feed_dict` 변수를 통해서 값을 넣어준다.
```python
print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))
```
결과는 다음과 같다.
```python
7.5
[ 3.  7.]
```
`feed_dict` 변수는 graph의 다른 tensor에 값을 넣어줄 때에도 사용될 수 있다. `run`할때 `tf.placeholder`에 값을 넣어주지 않으면 에러가 난다.

### Datasets
`Datasets`은 모델에 데이터를 넣어주기 위한 방법이다. 데이터로부터 `tf.Tensor`을 얻기 위해서, `tf.data.Iterator`로 변환해주어야 한다. 그 후에 이터레이터의 `get_next` method를 통해 데이터를 뽑아낸다.



