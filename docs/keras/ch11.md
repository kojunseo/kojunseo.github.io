# Ch11. ImageDataGenerator - flow, from_directory
* 데이터를 훈련하기 위해, 데이터 로더를 정의할 수 있다.
* 데이터를 원하는 형태로 주기 위해서, 혹은 데이터를 효율적으로 GPU, RAM에 올리기 위해 데이터 제너레이터를 활용한다.
> 실시간 데이터 증강을 사용해서 텐서 이미지 데이터 배치를 생성합니다. 데이터에 대해 (배치 단위로) 루프가 순환됩니다.

## 1. ImageDataGenerator
* 우선 데이터 생성기 객체를 정의해준다.
* [공식코드](https://keras.io/ko/preprocessing/image/)를 통해 공식 웹페이지를 확인해볼 수 있다. 자세한 파라미터 설명이 나온다.
```python
tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0,
    height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
    channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
    horizontal_flip=False, vertical_flip=False, rescale=None,
    preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None
)
```
* ImageGenerator는 데이터 Augmentation을 기본으로 제공한다. (i.e., flip, shift, zoom ..)

## 2. .flow(x, y)
* 만약 변수 x, y로 데이터를 미리 로드해놓았다면, .flow를 통해서 해당 데이터를 넣어주면 자동으로 해당 값을 augmentation하여 반환해준다.
* 미리 데이터를 전부 로드해야하는 상황일 경우 이 방법을 쓰면 좋다.

```python

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# 특성별 정규화에 필요한 수치를 계산합니다
# (영위상 성분분석 백색화를 적용하는 경우, 표준편차, 평균, 그리고 주성분이 이에 해당합니다)
datagen.fit(x_train)

# 실시간 데이터 증강을 사용해 배치에 대해서 모델을 학습합니다:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)
```

## 2. .flow_from_directory(directory)
* 만약 데이터를 폴더별로 잘 정리했다면 이 매서드를 활용할 수 있다.
* 폴더에 구성 그대로 데이터를 처리하여 로드해주는 강력한 함수이다.
* 그러나 모든 상황에서 데이터가 폴더별로 정리되어있지는 않기 때문에, 사용가능성이 매우 높다고 할 수는 없다.

```python
# 훈련용 제너레이터 객체 정의
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
# 테스트용 제너레이터 객체 정의
test_datagen = ImageDataGenerator(rescale=1./255)

# 훈련 폴더로 데이터 제너레이터 생성
train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
# 테스트 폴더로 데이터 제너레이터 생성
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```

## Tips
* 만약 데이터 제너레이터를 그냥 for문으로 호출하면, 이미지의 변형 버전이 배치 개수만큼 나온다. 
* 제너레이터는 이미지를 배치 개수만큼 리턴해주는 매써드라 생각하면 된다.