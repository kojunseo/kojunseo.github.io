# Ch5 Funtional Model of Keras
* Sequential Model의 경우, Sequential을 통해 자동으로 빌드를 하지만, Funtional의 경우 아니다.
* 총 두 가지의 스텝이 필요하다.
    + 레이어들을 output으로 이어준다.
    + 인풋과 아웃풋을 모델에 넣어 연결한다.
* 인풋과 아웃풋이 하나 이상일 때도 사용가능하기 때문에, 복잡한 모델 빌드에 용이하다.
* 함수를 통해 모델을 정의하고 함수를 호출하여 활용할 수 있다.

## Code Example
### 이전 단원에서 예시에 활용한 사용한 모델을 그대로 빌드해보자.
```python
import tensorflow as tf
inp = tf.keras.layers.Input([128])
x = tf.keras.layers.Dense(128)(inp)
out = tf.keras.layers.Dense(3)(x)

model = tf.keras.Model(inp, out)
```
* 보다시피 레이어의 output을 그다음 레이어에 넣어서 call한다.
* 그리고 마지막 모델 부분에서 inp, out 을 통해 최종 모델을 빌드하는 것을 볼 수 있다.
* 인풋 레이어를 정의하기 때문에, 다중 인풋을 정의할 수 있는 강점이 있다.
* 인풋 레이어가 정의되어, build과정이 필요없다.

### Multiple Input
```python
import tensorflow as tf
inp1 = tf.keras.layers.Input([128]) # 1번 인풋
x1 = tf.keras.layers.Dense(32)(inp1) # Dense를 통과시킨 output

inp2 = tf.keras.layers.Input([32]) # 2번 인풋
x2 = tf.keras.layers.Dense(128)(inp2) # Dense를 통과시킨 output

x = tf.keras.layers.Concatenate(axis=1)([x1, x2]) # 두 레이어의 출력을 통합
out = tf.keras.layers.Dense(3, activation="softmax")(x) # 결과 출력층

model = tf.keras.Model([inp1, inp2], out)
```

### Multiple Output (Multi-task Learning에 활용가능하다.)
```python
inp = tf.keras.layers.Input([128])
x = tf.keras.layers.Dense(128)(inp)
out1 = tf.keras.layers.Dense(3, activation="softmax")(x)
out2 = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inp, [out1,out2])
```

[예시코드](https://github.com/KorKite/study-keras-basic/blob/main/ch6/example.py)에서 실제 결과를 확인해보자.


## 자주 발생하는 에러!
> ValueError: This model has not yet been built. Build the model first by calling `build()` or calling `fit()` with some data, or specify an `input_shape` argument in the first layer(s) for automatic build.
* 만약 위와 같은 에러가 발생했다면 **tf.keras.Model**에 []를 넣어서 인풋과 아웃풋 레이어를 묶어준 경우이다. 풀어주자.
* model = tf.keras.Model([inp, out]) -> model = tf.keras.Model(inp, out)