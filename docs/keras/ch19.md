# Ch19 Special
* 위의 내용 외에 실제로 훈련시에 자주 활용되는 기법들을 공개합니다.

## Model Save, Load
* Model을 저장하고 불러와야하는 경우가 생깁니다.
* 저장은 Keras의 Callback중 체크포인트 저장을 통해서 할 수 있습니다.
* 혹은 다음과 같은 방법을 통해서도 가능합니다.
```python
model.fit(x, y)
model.save("/path/to/save/model.h5") # 모델 저장

import tensorflow as tf
loaded_model = tf.keras.models.load_model("/path/to/save/model.h5")
```
* 보통의 경우 h5로 저장합니다. 이렇게 저장하면 다른 세팅 없이 불러와 사용할 수 있습니다.

## Model Save, Load 2
* 그러나, Subclass API를 통해 복잡하게 모델을 정의했다면 이런 방법으로는 저장이 되지 않습니다.
* 가중치만 저장해서 모델에 불러오는 방식으로 진행해야 합니다.

#### Train.py
```python
model.fit(x, y)
model.save_weights("/path/to/save/model.ckpt") # 모델 저장
```
* 두개의 코드에서 나눠서 진행한다고 할 때, 우선 ckpt파일로 save_weight를 진행합니다.
* ckpt같은 확장자를 안써주면, 폴더 형식으로 저장이 됩니다. 이렇게 해도 됩니다.

#### Test.py or Inference.py
```python
model = ExModel() # 모델의 레이어를 정의하고
model.compile(...) # 컴파일까지 진행한 후
model.load_weights("/path/to/save/model") # 그곳에 가중치를 입력하는 방식
```
* 이 방식은 가중치만 저장했기 때문에, 가중치가 아닌 레이어나 옵티마이저 등은 따로 만들어주어야 합니다.
* 따로 정의해서 weights를 입력합니다.

## History Plot
* 훈련과정에서 Loss, val_loss 등을 저장해야하는데, 진행상황을 저장하고 싶은 경우가 생깁니다.
* 이 경우, History Plot을 그려서 저장해줍니다.
```python
history = model.fit(x, y) # fit을 할 때, history에 결과를 할당하면, 결과가 저장됩니다.

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy']) # 훈련시에 넣어준 metric의 변수명으로 동작합니다.
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show() # 로컬이나 주피터의 경우
plt.savefig("path/to/save/figure.png") # 서버의 python 환경
```


## More and More
* Special Lesson의 경우, 계속하여 추가됩니다. 
* 이 교재에 있는 내용을 알았으면, 그 이후는 본인이 모델을 사용해가면서 배우면 됩니다.
