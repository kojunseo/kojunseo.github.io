# Ch.3 - Basic Tensorflow
    기본적인 텐서플로우의 내용을 익혀봅시다.
    
## 1. Basic Tensorflow Tensor
* Vector는 2차원의 데이터를, Tensor는 3차원 이상의 데이터를 가르킨다.
* Tensorflow는 이 Tensor를 활용하여 쉽게 연산할 수 있도록 도와주는 프레임워크이다.
* Numpy와 유사하며, 넘파이에서 가능한 연산이 이 텐서로 모두 가능하다.

## 2. Tensorflow inner library: keras

```python
import tensorflow as tf
```

위의 코드를 통해 텐서플로우를 불러올 수 있다.
보통 tf라는 단축어로 tensorflow를 활용한다.

1. tf.keras
    + 파이썬으로 작성된 딥러닝 API로 텐서플로우 위에서 작동한다.
    + 간단하고 강력하며, 매우 안정적인 API에 속한다.
    + 텐서플로우의 모든 기능과 대부분 잘 붙는다.
    + tf1에서 tf2로 넘어오면서 torch의 코드와 매우 유사성을 가지게 되었다.

2. tf.keras.backend
    + tf... 하고 뒤에 붙는 Tensor연산을 keras.layer의 API와 활용할 수 있도록 재정의된 엔진이다.
    + expand_dims, lambda 등의 기능을 활용하기 위해 자주 활용된다.

### GPU Usage (Old)
* GPU가 달려있는 서버를 사용할 경우 텐서플로우는 최대한 많은 양의 GPU를 확보하려고 한다.
* 그렇기 때문에, GPU가 2대일 경우 한대만 활용해주어야 한다.
```python
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
```
다음 코드를 통해 특정 GPU만을 활용하도록 할 수 있다.

### GPU Usage 2 (New)
* (tf_alloc)[https://github.com/KorKite/tf_alloc]이라는 패키지를 활용하면 더 쉽게 특정 GPU를 특정 비율만큼만 할당할 수 있다.
```python
from tf_alloc import allocate
allocate(1, 0.5) # 1번 GPU를 0.5만큼만 사용하겠다는 코드
```
