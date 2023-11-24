# Ch5 Sequential Model of Keras
> A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
* 파이썬의 Sequential 모델의 경우, 인풋과 아웃풋이 하나씩 존재하는 모델에 레이어를 쌓아주기 적합한 방법이다.
* 간편하기 때문에 모델 내의 submodel을 만들어주기 위해서도 많이 사용된다.

## Code Example
```python
    import tensorflow as tf
    model = tf.keras.Sequential([
        # 필요한 레이어를 리스트 내에 넣어준다.
    ])
```
위와 같은 방법으로 하면 손쉽게 모델을 구성할 수 있다.
[코드 예시](https://github.com/KorKite/study-keras-basic/blob/main/ch5/example.py)를 통해 처음 모델을 빌드해보자.

## 확인해보기
```python
model.build([None, 128])
model.summary()
```
모델이 성공적으로 생성되었다면, 이제 input 크기를 지정해주면 그에 맞는 모델이 완성된다.
build를 통해 input의 크기를 지정해서 모델을 빌드한다.
그후 summary()를 수행하면 모델의 요약을 보여준다.