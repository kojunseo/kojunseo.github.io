# Custom Layer

## Why Custom Layer?
* Keras에 이미 많은 종류의 레이어가 정의되어 있지만, 모든 레이어가 정의되어 있지는 않다.
* 추가로 필요한 레이어가 존재할 때, 우리는 직접 레이어를 정의하고 활용할 필요가 있다.
* 혹은 레이어 하나로 반복되는 여러개의 레이어를 묶고 싶을 때도 활용할 수 있다.

## Template
```python
class LayerName(keras.layers.Layer): # Layer객체를 상속한다.
    def __init__(self, ...):
        super(LayerName, self).__init__()
        # 레이어에 필요한 가중치나 레이어를 정의할 수 있다.     

    def call(self, inputs):
        # 실제 데이터가 들어오고 나가는 흐름을 작성하면 된다.

        return 
```
* 기본적인 템플릿은 다음과 같다.
* 레이어 객체를 상속하고, init에는 필요한 레이어나 가중치를, call에는 실제 데이터의 흐름을 적으면 된다.

## Example1: Linear
* 다음은 실제 케라스에서 보여주는 Linear Layer의 예시이다.
```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```
* init에서는 필요한 가중치를 정의합니다. 가중치는 tf.Variable로 선언할 수 있습니다. (넘파이처럼 활용가능)
* call에는 데이터가 들어옵니다. 데이터가 들어와서 연산되는 과정을 보여줍니다. 가중치 갱신은 Layer객체가 자동으로 합니다.
* call은 파이썬 자체의 객체 정의 방식으로 
* 다음과 같은 객체 정의시 hello(init)(call)에서 init에 처음 넣는 값은 init으로, 그 뒤에 붙은 정의된 객체 호출은 call입니다.


## Example2: Attention
* 어텐션은 케라스 2.7.0에 추가되었습니다. 그 전까지는 어텐션이 없었기 때문에 정의해서 사용해주었었습니다.
```python
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        e = K.squeeze(e, axis=-1)   
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = K.sum(context, axis=1)
        return context
```

* build는 input_shape에 따라 가중치를 생성하는 것입니다.
* 보통은 미리 input_shape을 알지 못하기 때문에, 다음처럼 작성하는 편이 권장됩니다.