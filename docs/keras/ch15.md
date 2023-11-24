# Ch15 timedistributed layer

## What is timedistributed layer?
<img src="../../figures/mil-ex.png" width=800>

* 다음과 같은 모델을 설계한다고 해보자.
* 다음 모델은 25장의 이미지를 활용하여 이미지를 분류하는 모델이다.

### 작동 프로세스
1. 하나의 VGG16을 통해 각 이미지에 대한 피처를 추출한다. [1 x 512]사이즈
2. VGG피쳐 25개를 결합하여 [25 x 512]사이즈의 백터를 만든다.
3. 결합된 [25 x 512]사이즈의 백터를 활용하여 분류를 수행한다.

* 여기서 3번 프로세스는 이전과 동일한데, 1, 2번을 위한 모델을 어떻게 구성하면 좋을까?
* 여기서 활용가능한게 timedistributed layer이다.

## Code Example
```python
tf.keras.layers.TimeDistributed(
    layer, **kwargs
)
```
* 보기에는 매우 쉽다.

```python
inputs = tf.keras.Input(shape=(10, 128, 128, 3)) # 인풋이 128 x 128 x 3의 이미지가 10개 쌓여있는 모습이다.
conv_2d_layer = tf.keras.layers.Conv2D(64, (3, 3))
outputs = tf.keras.layers.TimeDistributed(conv_2d_layer)(inputs)
```
* 위에처럼 쌓아주면 [128 128 3] 이미지를 분류하는 Conv2D를 10번 거쳐서 벡터를 뽑아준다.

## Code Example - VGG16 Pretrained moel
* 만약 VGG16 사전학습 모델을 application으로 부터 로드해서 활용해주려면 어떻게 하면 좋을꺄?
* 아래는 실제 연구에 활용된 코드이다. (그림에서 활용된 코드)
```python
base_cnn = tf.keras.applications.VGG19(
    include_top = False,
    input_shape = (224, 224, 3),
    weights = None
)
intermediate_model = tf.keras.Model(
    inputs = base_cnn.input, outputs = base_cnn.output
)
input_layer = tf.keras.layers.Input(shape=(depth, 224, 224, 3))
timeDistributed_layer = tf.keras.layers.TimeDistributed( intermediate_model )(input_layer)
glob = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(timeDistributed_layer)
```
* 이렇게 코드를 짜주면 1개의 VGG에서 나오는 벡터는 [1 x 512]의 형태가 된다. (GAP)
* VGG16을 intermediate 모델을 정의하여 넣을 수 있다.
* 이렇게 해야 VGG16 모델을 수정하여 timedistributed layer에 넣어줄 수 있다.
* 이러면 3D Convolution을 사용한 것과 같은 벡터 크기를 처리해줄 수 있다.