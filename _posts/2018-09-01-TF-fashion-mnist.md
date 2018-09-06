---
layout: post
title: '[TensorFlow] Fashion-MNIST tutorial'
categories:
  - Coding
tags:
  - deep learning
  - tensorflow
  - implementation
  - tutorial
---

Reference : [TensorFlow Basic Classification](https://www.tensorflow.org/tutorials/keras/basic_classification)<br>
`TensorFlow`를 이용하여 Fashion-MNIST dataset을 가지고 classification task를 수행해보는 튜토리얼입니다. 좀더 기초적인 low-level API 가이드는 [TensorFlow  Guide](https://leejunhyun.github.io/coding/2018/09/01/TF-introduction.html)를 참조하시기 바랍니다.<br>

![Fashion-MNIST-01](/assets/img/TensorFlow/Fashion-MNIST-01.png)
*Fashion-MNIST dataset*

---
---
## Setup
```python
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
```
필요한 라이브러리들을 import 한다. `keras`를 사용하면 high-level 모듈을 불러와서 학습모델을 쉽게 만들 수 있다. `numpy`는 행렬연산을 위한 파이썬 라이브러리다. `matplotlib.pyplot`는 시각화를 위한 파이썬 라이브러리다.

---
## Import the Fashion MNIST dataset
Fashion-MNIST data는 28 x 28 픽셀의 저해상도 데이터셋이다. 기존의 MNIST 를 대체하고자 만들어졌으며, 10가지 클래스의 70,000장 이미지로 이루어져있다.

```python
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

60,000장의 train data와 10,000장의 test data를 나눠서 load한다.

```sh
Downloading data from http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
32768/29515 [=================================] - 0s 5us/step
Downloading data from http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
26427392/26421880 [==============================] - 7s 0us/step
Downloading data from http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
8192/5148 [===============================================] - 0s 0us/step
Downloading data from http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
4423680/4422102 [==============================] - 5s 1us/step
```
dataset 은 4개의 NumPy array로 반환된다. `train_images`,`train_labels`는 모델을 학습할때 사용되고, `test_images`,`test_label`은 모델 성능을 점검할 때 사용된다.

|Label|	Class|
|:----:|:---:|
|0|	T-shirt/top|
|1|	Trouser|
|2|	Pullover|
|3|	Dress|
|4|	Coat|
|5|	Sandal|
|6|Shirt|
|7|	Sneaker|
|8|	Bag|
|9|	Ankle boot|

```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

## Explore the data
```python
train_images.shape
```
```sh
(60000, 28, 28)
```

train label 또한 60,000개다.
```python
len(trian_labels)
```
```sh
60000
```
test도 같은 방법으로 확인하면 10,000개씩의 데이터가 있음을 알 수 있다.


## Preprocess the data
```python
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
```
`matplotlib.pyplot`을 이용하여 이미지중 하나를 열어본다. 
![Fashion-MNIST-02](/assets/img/TensorFlow/Fashion-MNIST-02.png)
`plt.colorbar()`은 픽셀값의 분포를 알수있게 해준다. 0에서 255값을 갖는 것을 알 수 있다. 뉴럴넷에 넣어주기 위해서 데이터를 [0,1]범위의 `float`형으로 변환해준다.
```python
train_images = train_images / 255.0

test_images = test_images / 255.0
```
학습 데이터셋에서 첫 25개의 이미지만 클래스이름과 함께 나타내본다.
```python
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
```
![Fashion-MNIST-03](/assets/img/TensorFlow/Fashion-MNIST-03.png)

## Build the model

본격적으로 모델을 만들고 컴파일한다.

#### Setup the layers
뉴럴넷의 기본적인 구성은 `layer`으로 쌓는다. `tf.keras.layers.Dense`와 같은 레이어들에는 학습 파라미터가 포함되어있다.
```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```
`model` 의 첫번째 레이어인 `keras.layers.Flatten`은 28 by 28 픽셀의 2d-array 데이터를 1d-array 의 784 (=28*28) 픽셀로 만들어준다. 이 레이어는 단순히 데이터를 재구성 하는 레이어이기때문에 학습 파라미터를 포함하지는 않는다.

데이터를 784 픽셀로 펴준 후에는 두 층의 `keras.layers.Dense`레이어를 통과시킨다. 첫번째 레이어는 128개의 노드와 `tf.nn.relu`로 이루어져있고, 두번째 레이어는 10개의 노드와 `tf.nn.softmax`로 이루어져있다. 마지막 `softmax`레이어는 출력값들의 합이 1인 확률값으로 만들어주는 함수이다. 최종 노드 10개는 각각 클래스에 해당하는 확률값을 내뱉게 된다.


#### Compile the model
모델을 학습하기 전에 `compile`에서 몇가지가 필요하다.
* Loss function - 학습을 하기 위해 최소화 시킬 대상인 `loss`를 정의한다.
* Optimizer - `loss`를 어떻게 최소화 시킬 것인지에 대한 방법을 선언한다.
* Metrics - 학습이나 테스트가 되는 동안 모니터링 할 수치를 정한다.

```python
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

## Train the model

모델을 학습할때는 다음과 같은 단계를 거친다.
1. 모델에 train 데이터(`train_images`,`train_labels`)를 넣어준다.
2. 모델이 image에 맞는 label을 내뱉을 수 있도록 학습한다.
3. test 데이터(`test_images`,`test_labels`)를 이용해 학습된 모델의 성능을 확인한다.

`model.fit`을 통해 학습을 시작한다.
```python
model.fit(train_images, train_labels, epochs=5)
```
```sh
Epoch 1/5
60000/60000 [==============================] - 2s 36us/step - loss: 0.4989 - acc: 0.8249
Epoch 2/5
60000/60000 [==============================] - 2s 34us/step - loss: 0.3787 - acc: 0.8634
Epoch 3/5
60000/60000 [==============================] - 2s 34us/step - loss: 0.3378 - acc: 0.8763
Epoch 4/5
60000/60000 [==============================] - 2s 35us/step - loss: 0.3141 - acc: 0.8859
Epoch 5/5
60000/60000 [==============================] - 2s 35us/step - loss: 0.2938 - acc: 0.8915
```
모델이 학습되면서 `loss`와 `accuracy`가 표기된다. 이 모델은 train 데이터에서 약 89%의 정확도를 보인다.

## Evaluate accuracy
학습된 모델의 성능을 test 데이터를 통해 확인한다.
```python
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
```
```sh
10000/10000 [==============================] - 0s 20us/step
Test accuracy: 0.8741
```
test 데이터에서의 정확도는 약 87% 정도 나오는 것을 확인할 수 있다. test 성능이 train 성능보다 많이 떨어질 때는 *overfitting*의 여지가 있다.


## Make predictions
모델을 학습할 때, 특정 이미지에 대해 output(prediction)을 뽑아볼 수도있다.
```python
predictions = model.predict(test_images)
```
이제 `predictions`에는 `test_images`에 대한 model output들이 저장되어 있다.
```python
predictions[0]
```
첫번째 이미지에 대한 prediction을 보면 아래처럼 나온다.
```sh
array([1.2189614e-05, 9.8493892e-08, 2.6474936e-06, 4.6750273e-08,
       2.2893637e-07, 4.9046555e-04, 4.9265759e-06, 9.2500690e-03,
       2.6400221e-05, 9.9021286e-01], dtype=float32)
```
총 10개의 값으로 이루어져있으며, 각각 클래스에 대한 확률값을 나타낸다. 몇번째 인덱스가 가장 높은 값을 갖는지 보려면 `np.argmax`를 사용한다.
```python
np.argmax(predictions[0])
```
```sh
9
```
아홉번째 값이 가장 높다는 것을 알 수 있다. 즉, 이 모델은 `test_images[0]` 이미지에 대해 아홉번째 클래스라고 예측한 것이다. 정답이 맞는지 아래와같이 확인할 수 있다.
```python
test_labels[0]
```
```sh
9
```

결과를 시각화하기위한 함수를 정의한다. 시각화에 대한 방법은 각자 정의하기 나름이다. 여기서는 input 이미지와, 아래엔 예측과 확률값, 정답을 글자로 출력하고 오른쪽엔 출력값을 그래프로 나타낸다.
```python
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
```
방금 봤던 0번 데이터에 대해 확인해보면, 다음과 같다.
```python
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
```
![Fashion-MNIST-04](/assets/img/TensorFlow/Fashion-MNIST-04.png)
```python
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
```
틀린경우엔 아래처럼 표기된다.
![Fashion-MNIST-05](/assets/img/TensorFlow/Fashion-MNIST-05.png)
마지막으로, 하나의 이미지에 대해 출력값을 뽑아내본다. `keras`는 배치단위의 입력에 최적화되어있다. 즉, `train_images`는 60,000장의 이미지가 쌓여서 (60000,28,28)의 입력이었고, 이중에서 배치 사이즈만큼씩 모델에 들어간다. 그러나 한장의 이미지의 차원을 뽑아보면,
```python
img = test_images[0]

print(img.shape)
```
```sh
(28,28)
```
2차원으로 나오는 것을 알 수 있다. 이를 여분의 차원을 늘려주어야 한다.
```python
img = (np.expand_dims(img,0))

print(img.shape)
```
```sh
(1,28,28)
```
이제 모델에 넣어줄 수 있다.
```python
predictions_single = model.predict(img)

print(predictions_single)
```
```sh
[[1.2189591e-05 9.8493892e-08 2.6474886e-06 4.6750095e-08 2.2893614e-07
  4.9046462e-04 4.9265759e-06 9.2500662e-03 2.6400170e-05 9.9021286e-01]]
```
```python
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
```
![Fashion-MNIST-06](/assets/img/TensorFlow/Fashion-MNIST-06.png)

`model.predict`또한 배치를 고려해서 한장의 이미지여도 여분의 차원을 만들어  output을 내므로, 결과를 볼때는 [0]인덱스를 봐야한다.
```python
np.argmax(predictions_single[0])
```
```sh
9
```
