# Ch9. Model Training 1 - compile, fit
## Model Fit
* 케라스에서는 fit을 통해 훈련을 매우 쉽게 가능하도록 했다.
* 복잡한 일련의 과정없이 compile 후 Fit 만 해주면 된다.

## 0. Load Data
* 훈련에 앞서서 데이터를 준비해주어야 한다.
* 가장 기본적으로 데이터는 Numpy 배열로 이뤄진 데이터로도 학습이 가능하다.
* 훈련은 두개의 데이터 쌍으로 이뤄지는데 두개 모두 [배치 크기, 데이터]의 형태를 이룬다.
* 만약 이미지를 분류하는 테스크를 훈련한다고 할 때, 넘파이의 경우
    + 훈련데이터 = [데이터 개수, 가로크기, 세로크기, 채널] 
    + 테스트데이터 = [데이터 개수, 라벨] 
    + 형태의 어레이 크기로 준비해야한다. (**데이터 개수는 훈련, 테스트 동일해야함**)
    + 만약 이미지가 아닐 경우 데이터 개수 뒤에 해당 데이터 크기를 써주면 된다.
* 훈련 테스트가 동일해야하는 이유는, 한개의 데이터에 대해 정답과의 연결을 하는 것이기 때문이다.
* 모든 테스크가 그러진 않지만 많은 양의 테스크가 위의 규칙을 따른다.
* 세그멘테이션의 경우 라벨이 이미지일 수도 있다.

## 1. Load model
* ch6에 예시에서 만든 간단한 모델을 불러와보겠다.
```python
import tensorflow as tf
from ch6.example import model

my_model = model()
```
* 이 모델은 input 크기가 128이다. 그렇다면 실제 준비해야하는 인풋 데이터의 크기는
    + [데이터개수, 128] 형태이여야 한다.
* 이 모델의 output 크기가 3임으로 이것은 3개를 분류해주기 위한 분류 테스크임을 알 수 있다.
    + [데이터개수, 라벨] 형태로 아웃폿 데이터를 준비한다.

## 2. Compile  Model
* 다음은 compile을 해주어야 한다. 아래를 보자.
### Compile Parameters
```python
compile(
    optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, 
    target_tensors=None
)
```
* 이것이 compile에 들어가야 하는 파라미터이다.

### 실제 compile 예시
```python
import tensorflow as tf
from ch6.example import model

my_model = model()
my_model.compile(
    optimizer="adam", 
    loss="categorical_crossentropy", 
    metrics=["acc"]
)
```
* 다른 거 없이, opimizer, loss와 metric 이 3가지만 지정해서 훈련을 진행한다.
* loss에서 만약 one hot vector를 사용할 경우 categorical을
* 일단 숫자 라벨을 쓸 경우 sparse_categorical_crossentropy를 사용하면 된다.

## 3. Fit the model
* fit함수에 필요한 파라미터는 다음과 같디.
```python
fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, 
    callbacks=None, validation_split=0.0, validation_data=None, 
    shuffle=True, class_weight=None, sample_weight=None, 
    initial_epoch=0, steps_per_epoch=None, validation_steps=None)
```
### 중요한 파라미터 정리
* x = 훈련용 x값
* y = 훈련용 y값
* batch_size = 몇 배치만큼 훈련할 것인지.
* epochs = 몇번 반복하여 모델을 훈련할 것인지
* callbacks = 훈련 과정에서 중간에 결과나 모델을 가지고 무언가 할 수 있도록 하는 것
* validation_split = 훈련용 x에서 일부분 비율만큼을 때서 검증용으로 사용할 수 있도록 비율을 지정한다.
* shuffle = 데이터의 순서를 중간중간 섞어준다.
* validation_data = split하지 못하거나 사전에 정의된 경우 별도의 데이터를 지정한다.

## 4. save the model
* 모델을 훈련했으면, 나중에 이 모델을 다시 꺼내올 수 있도록 저장하고, 불러올 수 있어야 합니다.
```python
model.save("[저장하고 싶은 경로].h5")
model = tf.keras.models.load_model("[저장한 모델 경로].h5")
```
* 기본적인 모델은 이런 방식으로 저장 가능합니다.
* 추가적인 수정이 된 레이어를 쓸 경우, ch19의 커스텀 레이어 부분을 참고해주세요!


### 실제 fit 사용예시
* model이 준비되었고, 데이터는 100개의 x, y가 주어졌다고 가정하자.

```python
import tensorflow as tf
from ch6.example import model
x = np.ones([100, 128])
y = np.ones([100, 1])
my_model = model()
my_model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=["acc"]
)
my_model.fit(
    x = x, y = y,
    epochs=100,
    validation_split=0.3,
    batch_size=16
)
```
* 이처럼 하면 자동으로 모델이 훈련을 진행한다.
* [코드 예시]를 통해 mnist예제를 분류하는 모델을 만들어보자.