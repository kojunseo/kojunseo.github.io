# ch4 - Basic Keras Layers

## Linear Based Layers
### Dense Layer (Fully Connected Layer)
    tf.keras.layers.Dense(
        units, activation=None, use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None, **kwargs
    )
단순한 Linear Layer로 **y = wx + b** 의 단순 연산을 수행

<img src="./../../figures/dense.png" width=500>


## Convolution Based Layers
Convolution 연산을 수행하는 레이어이다. 컨볼루션 연산을 모른다면 구글을 참고!
### 1. 1D Convolution Layer
    tf.keras.layers.Conv1D(
        filters, kernel_size, strides=1, padding='valid',
        data_format='channels_last', dilation_rate=1, groups=1,
        activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None, **kwargs
    )
<img src="./../../figures/1d_convolution.png" width=500>


### 2. 2D Convolution Layer
    tf.keras.layers.Conv2D(
        filters, kernel_size, strides=(1, 1), padding='valid',
        data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
        use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None, **kwargs
    )
<img src="./../../figures/2d_convolution.png" width=500>


### 3. 3D Convolution Layer
    tf.keras.layers.Conv3D(
        filters, kernel_size, strides=(1, 1, 1), padding='valid',
        data_format=None, dilation_rate=(1, 1, 1), groups=1, activation=None,
        use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None, **kwargs
    )
<img src="./../../figures/3d_convolution.gif" width=500>

### 4. ConvTranspose (UpConv, DeConv)
Convolution을 반대로 해주는 것이다. 오토인코더의 디코더 부분, GAN의 Generator, Unet 등에 활용된다.

#### 1. Upconv 1D
    tf.keras.layers.Conv1DTranspose(
        filters, kernel_size, strides=(1, 1), padding='valid', ...
    )

#### 2. Upconv 2D
    tf.keras.layers.Conv2DTranspose(
        filters, kernel_size, strides=(1, 1), padding='valid', ...
    )

#### 3. Upconv 3D
    tf.keras.layers.Conv3DTranspose(
        filters, kernel_size, strides=(1, 1), padding='valid', ...
    )
<img src="./../../figures/deconv.png" width=500>

### Upsampling VS ConvTranspose
두개의 차이가 햇갈린다면 아래의 인용구를 참고!
> UpSampling2D is just a simple scaling up of the image by using nearest neighbour or bilinear upsampling, so nothing smart. Advantage is it's cheap. Conv2DTranspose is a convolution operation whose kernel is learnt (just like normal conv2d operation) while training your model.
> = Upsampling은 단순히 스케일 업해서 만든걸로 파라미터 X / ConvTranspose는 실제 역-컨볼루션 연산으로 파라미터 O

***

## Pooling Layers
### 1. Maxpooling (1D ~ 3D)
    tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=None, padding='valid'
    )

### 2. Avgpooling (1D ~ 3D)
    tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2), strides=None, padding='valid'
    )
<img src="./../../figures/avg-max-pool.png" width=500>


### 3. GlobalAveragePooling (= Flatten 대신 활용하는 레이어)
    tf.keras.layers.GlobalAveragePooling2D()

<img src="./../../figures/gap.png" width=500>

***

## Recurrent Neural Net Based Layers
 입력과 출력을 시퀀스 단위로 처리하는 모델. 
 ### 1. RNN
    tf.keras.layers.RNN(
        cell, return_sequences=False, return_state=False, go_backwards=False,
    )

### 2. GRU
    tf.keras.layers.GRU(
        units, activation='tanh', recurrent_activation='sigmoid',
        use_bias=True, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False,
        go_backwards=False,
    )

### 3. LSTM
    tf.keras.layers.LSTM(
        units, activation='tanh', recurrent_activation='sigmoid',
        use_bias=True, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False,
        go_backwards=False,
    )

<img src="./../../figures/rnn-based.png" width=500>

## Layers etc.
### 1. Input Layer
    tf.keras.layers.Input(
        input_shape=None
    )
Funtional API 사용시 Input 텐서를 정의하기 위한 레이어

### 2. Flatten
    tf.keras.layers.Flatten()
LSTM, CONV에서 나온 피쳐맵을 Dense에 넣기 전 펴주기 위한 레이어

### 3. Dropout
    tf.keras.layers.Dropout(
        rate
    )
특정 비율만큼 셀을 꺼서 훈련이 되지 않도록 하여 오버피팅을 방지하는 레이어

### 4. Bidirectional
    tf.keras.layers.Bidirectional(
        layer
    )
양방향 LSTM을 만들어주기 위한 것으로, 앞 뒤의 sequence를 참조할 수 있도록 해줌

### 5. Concatenate
    tf.keras.layers.Concatenate(
        axis=-1
    )
두 개의 레이어를 결합해준다. 한 개의 shape은 일치해야 결합이 가능함.

### 6. Add, Multiply
    tf.keras.layers.Add()
    tf.keras.layers.Multiply()
두 레이어 간의 곱샘, 덧샘 등의 연산을 수행할 수 있도록 해줌

### 7. Dot
    tf.keras.layers.Dot(
        axes,
    )
두 레이어 출력 간의 행렬연산을 수행하도록 해줌.