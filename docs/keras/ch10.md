# Ch10 Callbacks
* Callback은 fit으로 훈련하는 중간에 다양한 기능을 수행하도록 하는 역할을 수행합니다.
* 모델을 저장하거나, history를 그린다거나 하는 등의 역할입니다. 
* 이번에는 두개의 유명한 model_checkpoint, early_stop 콜백을 배워봅니다.

## 1. model_checkpoint
[Model Checkpoint](https://keras.io/api/callbacks/model_checkpoint/)에 방문하여 모델 체크포인트가 뭔지 볼 수 있습니다.
```python
tf.keras.callbacks.ModelCheckpoint(
    filepath,
    monitor="val_loss",
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    options=None,
    **kwargs
)
```
* 간단하게 설명하자면, 모델 체크포인트는 모델 파일을 특정 조건에 따라 저장하는 것입니다.
### Parameters
* filepath = 저장할 파일 경로입니다.
* monitor = 어떤 값을 기준으로 할지 정합니다. 보통 metric에 특정 값에 따라 저장되도록 합니다.
* verbose = 저장 시 안내문구를 출력할지 여부를 묻습니다.
* save_best_only = 제일 좋은 값만 저장합니다.
* save_weight_only = 모델의 레이어를 제외한 가중치만을 저장합니다. (나중에 모델을 다시 만들고 그 위에 가중치를 불러와야합니다.)
* model = 작을 때 저장할지 클때 저장할지 설정할 수 있습니다. ["min", "max"]


## 2. Early Stop
[Early Stop](https://keras.io/api/callbacks/early_stopping/)에 가서 자세한 내용을 볼 수 있습니다.

```python
tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=0,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)
```
* 간단하게 설명하자면, 모델의 성능이 올라가지 않을때 모델의 학습을 중간에 중단시키는 코드입니다.

### Parameters
* monitor = 어떤 값을 기준으로 저장할지 여부입니다.
* patience = 몇번의 에포크동안 개선이 없을 때 멈출지 정합니다.
* mode = 작은값을 혹은 큰값을 기준으로 할지 정합니다.

> 이 콜백을 원하는 것으로 하고 싶은 경우 커스텀 콜백 부분을 참고하시면 됩니다.