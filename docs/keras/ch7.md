# Ch7. Subclass API Method
* 모델을 정의하는 마지막 방법으로 keras의 모델클래스에서 상속을 받아 정의하는 방법이 있다.
* 뒤에 배울, tf.function을 통해 fit 자체의 수정이 가능해 용이하다.
* 가중치가 학습된 layer를 모델에서 그대로 불러올 수 있다는 장점 때문에, 모델의 일부분만 사용하는 경우에 자주 사용된다.

## 1. 상속
```python
import tensorflow as tf
class customModel(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        ...
```
* 위와 같이 class로 모델을 정의하고, 미리 정의된 Model을 상속받는다.

## 2. block define
```python
import tensorflow as tf
class customModel(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3,3)),
            tf.keras.layers.Conv2D(64, (3,3)),
            tf.keras.layers.MaxPooling2D(),
        ])
        self.block2 = tf.keras.layers.Dense(3)
```
* \__init__ 부분에 다양한 방법으로 모델의 레이어나 서브 모델을 정의할 수 있다.
* 이 정의된 모델 혹은 서브모델을 활용한다.

## 3. call function
```python
import tensorflow as tf
class customModel(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3,3)),
            tf.keras.layers.Conv2D(64, (3,3)),
            tf.keras.layers.MaxPooling2D(),
        ])
        self.block2 = tf.keras.layers.Dense(3)

    def call(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x
```
* 이제 call객체 함수에 x를 파라미터로 받아 이것을 funtional API로 정의하듯 처리하고 x를 리턴한다.
* 이렇게 하고 build를 해주면 모델 정의는 끝이다.
* 이제 여기에 새로운 객체 함수를 넣어 모델을 커스텀할 수도 있다. 이는 [autoencoder예제](https://github.com/KorKite/study-keras-basic/blob/main/applications/autoencoder.py)를 통해 확인해보자.


# Sum up
* ch5, 6, 7을 통해 우리는 세가지 다른 방법으로 모델을 정의했다.
* 연구나 실제 개발에 필요한 방법을 적절하게 활용하면 좋다.
* torch와 유사한 방법은 class로 활용하는 방법이다.
* 개인적으로 커스텀이 제일 용이한 것은 functional API이다.
* 빠르게 모델을 빌드하고 싶으면 sequential이다.
* 세가지 방법의 경우, 문서를 보지 않고도 작성할 수 있도록 해야한다.

## Code Application
+ 실제 위의 코드를 적용한 코드를 제공한다.
+ 이미지 분류 모델인 VGG16을 3가지 다른 방법으로 정의했다. 모두 기능적으로 동일하다.

### 1. Sequential API
[Sequential 코드예제](https://github.com/KorKite/study-keras-basic/blob/main/applications/vgg16_sequential.py)
### 2. Funtional API
[Funtional 코드예제](https://github.com/KorKite/study-keras-basic/blob/main/applications/vgg16_functional.py)
### 3. Class Inherit API
[Class 코드예제](https://github.com/KorKite/study-keras-basic/blob/main/applications/vgg16_class.py)
