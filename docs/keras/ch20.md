# What is special session?
## Batch size
```python
model.fit(x, y, batch_size = 32, epochs = 10)
```
* fit함수는 배치사이즈를 파라미터로 받는다.
* 배치사이즈는 몇 개의 샘플로 가중치를 갱신할 것인지를 설정하는 것이다.
* 만약 배치사이즈가 20이라면 전체 데이터에서 20개 만큼의 샘플만으로  가중치를 계산하는 것이다.

## Train Step
* 그렇다면 train step은 무엇일까?
* batch_size를 20이라고 하고 전체 데이터를 100개라 하면, 5번의 train step을 거친다.

## Epochs
* 100개의 데이터를 훈련할 때, 20의 배치사이즈로 5번 훈련 스탭을 통해 모든 데이터에 대해 가중치를 계산했을 때를 1개의 epoch이라 한다.