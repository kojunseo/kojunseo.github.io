# Ch18. Custom Loss, Metrics

## Custom Loss Function
* Loss Function은 모델의 예측값과 실제값의 차이를 정의하여 모델이 학습될 수 있도록 하는 함수입니다.
* 대부분의 Loss가 정의되어 있지만, 최신의 논문에 등장하는 로스 함수의 경우 정의를 해주어야 합니다.

### 기본적인 Loss 함수
```python
import keras.backend as K

def root_mean_squared_error(y_true, y_pred):
    err = y_pred-y_true
    square = K.square(err)
    mean = K.mean(square)
    root = K.sqrt(mean)
    return root
```
* 기본적으로는 함수처럼 선언하면 Loss를 만들어줄 수 있습니다. 
* y_true, y_pred를 받아서 계산하면 됩니다.
* Keras 모델 학습시에 사용할 경우 backend로 작성해야합니다. tf.으로 작성하면 훈련에 사용되지 못합니다.

### Subclass API Loss 함수
```python
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss

class root_mean_squared_error(Loss):
    def __init__(self, variable):
        ...

    def call(self, y_true, y_pred):
        ...
        return loss
```
* 만약에 loss 계산시에 특별한 변수를 추가해서 계산해야 한다면 다음과 같이 서브클래스로 정의할 수 있습니다.
* DiceCoefficient 와 같은 로스 작성시에 smooth를 변수로 주면 좋습니다.


```python
class DiceCoeffLoss(Loss):
    def __init__(self, smooth):
        self.smooth = smooth

    def call(self, y_true, y_pred):
        numerator = 2. * K.sum(y_true * y_pred)
        denominator = K.sum(y_true + y_pred)
        return K.mean(1 - numerator / (denominator + self.smooth))
```

## Metrics?
* Loss처럼 실제값과 예측값의 차이를 보고하지만, 이를 가중치 갱신에 활용하지는 않습니다.
* 대부분의 Metric은 보통은 케라스에 정의가 되어있지만, 새로운 메트릭을 정의해서 사용해야하는 경우도 있습니다.

### How to Use?
* Loss에서 Functional한 방법을 사용하는 것과 동일하게 사용할 수 있습니다.

```python
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
```
* 매우 쉽죠?
