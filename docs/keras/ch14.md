# GradientTape
* Tensorflow 2에 처음 등장한 방법으로, 자동미분을 도와주는 텐서플로우 라이브러리 입니다.
* 동적으로 Gradient값을 확인해볼 수 있습니다.
* fit의 경우, 세부 미분과정을 확인해볼 수 없지만 GradientTape의 경우, 과정을 확인하고 컨트롤 할 수 있음.
* fit의 경우, 빠르게 훈련이 가능하지만 세부 동작 원리를 몰라도 사용가능하기 때문에, 실제 딥러닝에 대한 깊은 이해 없이도 훈련할 수 있다.
* 딥러닝 공부를 위해선 GradientTape이 도움이 될 것이다. 
* GradientTape의 경우 토치의 스타일과 매우 유사하게 사용가능한 방법론이다. 

## How To use
* 우선 모델이 빌드 되어있고, ch11, 12, 13에서 배운 데이터로더가 구비되어있으며, optimizer, loss 등의 정의가 완료된 상태임을 가정한다.

```python
# Loss Function을 변수로 정의
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
# Optimizer를 변수로 정의
optimizer = tf.keras.optimizers.Adam()
# Metric 을 변수로 정의, Train, Test 분리
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
test_acc = tf.keras.metrics.SparseCategoricalAccuracy() 

# Loss 정의
train_loss = tf.keras.metrics.Mean()
test_loss = tf.keras.metrics.Mean()
```

* 위와 같이 필요한 함수들이 정의 되었다면 이제 GradientTape을 제대로 활용해보자

## Train Step
```python
@tf.function
def train_step(images, labels):
    # 미분을 위한 GradientTape을 적용합니다.
    with tf.GradientTape() as tape:
        # 1. 예측 (prediction)
        predictions = model(images)
        # 2. Loss 계산
        loss = loss_function(labels, predictions)
    
    # 3. 그라디언트(gradients) 계산
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # 4. 오차역전파(Backpropagation) - weight 업데이트
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # loss와 accuracy를 업데이트 합니다.
    train_loss(loss)
    train_acc(labels, predictions)
```
* 위와 같은 방법으로 한개 step에 대한 훈련이 진행된다.
* Train Step이 뭔지 모른다면, [다음을 참고하세요](https://github.com/KorKite/study-keras-basic/tree/main/contents/special-session)

### One By One
```python
@tf.function
def train_step(images, labels):
```
* 맨 위부터 살펴보자. @는 데코레이터로 파이썬에서 함수 안 함수 구조를 만드는데 활용된다. 데코레이터 강의가 아님으로 이는 넘어간다.
* tf.function은 쉽게 말하면 데코레이터 아래의 함수를 GPU위에 올려서 돌아가게 하겠다는 의미이다.
* 아래의 함수를 GPU에서 돌릴 수 있도록 컴파일 하는 녀석이다.

```python
    with tf.GradientTape() as tape:
        # 1. 예측 (prediction)
        predictions = model(images)
        # 2. Loss 계산
        loss = loss_function(labels, predictions)
```
* with GradientTape안에서 수행하는 모든 연산을 tape 변수 안에 자동으로 기록한다.
* 이제 tape에 loss와 예측내용이 저장되어 있다.


```python
 # 3. 그라디언트(gradients) 계산
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # 4. 오차역전파(Backpropagation) - weight 업데이트
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # loss와 accuracy를 업데이트 합니다.
    train_loss(loss)
    train_acc(labels, predictions)
```
#### Gradient 계산
* tape에 기록된 loss를 바탕으로 gradient 연산을 수행해 gradients 변수에 기록합니다.
* model.trainbale_variables는 모델 내부의 훈련가능한 변수들(레이어들)로 이 변수들에 대한 미분값을 계산합니다.

#### 오차역전파 weight update
* optimizer는 이제 이 Gradient를 바탕으로 모델을 효과적으로 업데이트 하는 역할을 수행합니다.
* 이 과정에서 각 gradient에 있는 값들과 모델의 훈련가능한 변수들 간에 업데이트가 진행됩니다.
* train_loss, train_acc는 훈련 loss와 정확도로, 함수에 훈련과정이 저장됩니다.


## Test Step
```python
@tf.function
def test_step(images, labels):
    # 1. 예측 (prediction)
    predictions = model(images)
    # 2. Loss 계산
    loss = loss_function(labels, predictions)
    
    # loss와 accuracy를 업데이트 합니다.
    test_loss(loss)
    test_acc(labels, predictions)
```

* Train을 봤다면 Test역시 비슷한 메커니즘임으로 쉽게 이해할 수 있을 겁니다.
* gradientTape이 없는 이유는 test시에는 미분이 필요가 없기 때문입니다.


## Epoch Process
```python
EPOCHS = 5

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)
        
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = '에포크: {}, 손실: {:.5f}, 정확도: {:.2f}%, 테스트 손실: {:.5f}, 테스트 정확도: {:.2f}%'
    print (template.format(epoch+1,
                           train_loss.result(),
                           train_acc.result()*100,
                           test_loss.result(),
                           test_acc.result()*100))
```
* 에포크가 5개라고 가정할 때, 실제로 for문을 통해서 에포크를 진행시킵니다.
* train_ds의 경우 훈련용 데이터로더, test_ds의 경우 테스트용 데이터로더입니다.
* for문을 통해 데이터로더에서 배치만큼 데이터를 꺼내와 images, labels에 순차적으로 담습니다.
* 그리고 방금 정의한 train_step, test_step에 넣게 되면, GPU위에서 위의 연산이 자동으로 수행됩니다.

* 마지막으로는 loss, acc를 담아둔 함수에서 결과를 뽑아 프린트합니다.

## tf.function
```python
@tf.function(experimental_relax_shapes=True)
```
* tf.function에 다음과 같은 파라미터를 줄 수 있습니다. 
* 이는 인풋을 처리할 때, 각 입력에 특화되지 않도록 변경해줌으로서 성능을 크게 향상 시킵니다.
* 거의 무조건 다음과 같은 파라미터를 사용해주면 좋습니다.

## Print Inside of tf.function
```python
@tf.function(experimental_relax_shapes=True)
def train_step():
    ...
    tf.print(things_to_print)
```
* tf.function을 데코레이터로 한 함수에서는 일반적인 print로는 실행되지 않음.
* tf.print를 사용하면 언제나 프린트가 가능하기 때문에, 이를 활용해주면 됨

## Application
* 이 과정을 가장 유용하게 활용하는 경우는 GAN 학습시입니다.
* GAN은 Loss가 2개 이상인 경우가 많습니다. 그럴때는 GradientTape을 2개 사용해서 한개에 모델에 대해 다른 로스를 적용할 수 있습니다.
* [텐서플로우 Pix2pix GAN 예제](https://www.tensorflow.org/tutorials/generative/pix2pix?hl=ko)를 통해 GradientTape을 최대한 잘 활용한 예시를 확인해보세요.

