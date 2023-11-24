# Ch.1 Basic Functions
## 0. Keras vs Tensorflow?
    1. 케라스는 파이썬으로 작성된 고수준의 신경한 API
    2. Tensorflow와 결합성이 높고, 아이디어를 최대한 빠르게 모델로
    구현하는 것을 목표로 하는 라이브러리입니다.


## 1. Activations
    1. 뉴런의 전기신호를 보내줄지 말지를 결정하는 모델을 본따서 만든 Step Function
    2. 그러나, 이 step function은 미분이 불가능하기 때문에 가능한 것으로 수정 필요
    3. 그래서 등장한 것이 이 Activation function임.

### 관련 자료
[관련자료1: keras activations](https://keras.io/ko/activations/)

[관련자료2: activation의 역사](https://89douner.tistory.com/22)



## 2. Metrics
    1. 모델을 평가하기 위해 사용됩니다.
    2. 분류, 회귀 크게 2가지의 테스크를 평가할 때 활용됩니다. 

### 관련 자료
[관련자료1: Keras Metrics](https://keras.io/api/metrics/)



## 3.Optimizers
    1. 경사하강법 통해 데이터를 훈련할 때, 어떤 방식으로 훈련 방식입니다.
    2. 뒤에 배울 모델을 compile하는 과정에서 필요한 두개의 파라미터 중 하나입니다.
    
### 관련 자료
[관련자료1: 각종 경사하강법 수식과 설명](https://ruder.io/optimizing-gradient-descent/index.html#rmsprop)

[관련자료2: Kears Optimizers](https://keras.io/ko/optimizers/)


## 4. Loss Function
    1. 대입한 결과와 실제 정답간의 간격을 의미한다.
    2. 모델 훈련 시 이 간격을 줄이는 방향으로 학습이 진행된다.
    3. 뒤에 배울 모델을 compile하는 과정에서 필요한 두개의 파라미터 중 하나입니다.

### 관련 자료
[관련자료1: Tensorflow Loss Functions](https://www.tensorflow.org/api_docs/python/tf/keras/losses)

[관련자료2: Keras Loss Function](https://keras.io/ko/losses/)