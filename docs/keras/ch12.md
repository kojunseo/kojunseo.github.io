# Ch12. Custom Data Loader 1 - Pre-load Data

* Ch11까지 잘 따라왔다면 이제 모델을 훈련하는 것은 어렵지 않아야 한다. 이제부터 기본적이지 않은 내용을 다뤄보자.

## Pre-knowledge
* 컴퓨터에서 훈련을 진행할 때, 데이터의 흐름을 생각해보자.
    + 1. 데이터가 Disk에 저장되어 있다.
    + 2. 데이터를 램에 올린다.
    + 3. 램에서 데이터를 처리한다.
    + 4. 램에서 데이터를 GPU에 전달한다.
    + 5. GPU에서 모델을 훈련한다.
* 위의 과정으로 모델의 훈련이 진행된다.
* 그러나 여기서 두가지 부분에서 데이터의 흐름이 어색해진다.(보통 용량은 Disk > RAM > GPU 순서이다.)

### 문제가 되는 두가지 가정
1. 만약 데이터가 RAM에 다 안올라가면 어떻하지?
2. 만약 RAM의 데이터가 GPU에 안올라가면 어떻하지?

* 이 두가지 문제를 해결해야한다.
* 2번의 경우 Batch_size로 일부만큼만 GPU에 올려 나눠서 학습하면서 해결할 수 있다. ([만약 에포크, 배치사이즈 등의 개념을 모른다면 클릭](https://github.com/KorKite/study-keras-basic/tree/main/contents/special-session) )
* 그러나, 1번의 경우 어떻게하면 좋을까?

### Solutions
* 이 문제를 해결하려면 RAM에 배치만큼만 데이터를 할당했다가 해제했다가 하면서 RAM이 초과되는 문제를 해결하면 될 것이다.
* 이 문제 이외에도 만약 데이터를 원하는 형태로 전달해주려면 어떻게할지에 대한 고민도 필요하다.

## Custom Dataloader
* 우리는 데이터로더를 커스텀해서 이와같은 문제를 해결할 수 있다.
* ch11에서 학습한 ImageDataGenerator같은 데이터로더를 직접 구성할 수 있다.

## Custom Dataloader - On first load
```python
from tensorflow.keras.utils import Sequence
class CustomDataloader(Sequence):
	def __init__(self, x_set, y_set, batch_size, shuffle=False):
	    self.x, self.y = x_set, y_set
	    self.batch_size = batch_size
	    self.shuffle=shuffle
```
* 만약 데이터가 RAM에는 모두 올라갈 수 있다고 한다면, 다음과 같이 Sequence를 상속받아 데이터 로더를 꾸릴 수 있다.
* 이제 데이터로더의 나머지 부분을 짜보자.
* 세 가지 필수 함수가 필요하다. (\__len__, \__getitem__, on_epoch_end)

### \__len__(self)
* 데이터로더의 전체 길이를 반환하는 함수이다.
* 500개의 데이터에서 배치사이즈가 100이라면 데이터 사이즈는 5가 되는 것입니다.
* 다음처럼 정의해주면 됩니다.
```python
import math
...
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)
```
### on_epoch_end(self):
* 매 애포크가 끝날때마다 처리해 줄 내용을 작성한다.
* 보통 shuffle이 필요할 때, 데이터를 섞는 용도로 많이 사용한다.
```python
def on_epoch_end(self):
  self.indices = np.arange(len(self.x))
  if self.shuffle == True:
      np.random.shuffle(self.indices)
```
self.indices는 x의 개수만큼을 정하고 이것을 섞어준다.


### \__getitem__(self, index)
* 이 함수가 메인다.
* index파라미터에 따라 해당하는 배치의 데이터를 반환한다.
* 다음 그림을 통해 getitem을 이해해보자.
<img src= "./../../figures/getitem.png" width = 500>

```python
def __getitem__(self, index):
    indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
    return self.x[indices], self.y[indices]
```
* index를 활용해서, 그림처럼 인덱스에 해당하는 데이터를 불러와 리턴해주면 된다.
* 본인이 원하는 형태로 전달하고 싶으면 이 곳에서 수정할 수 있다.

### 다음시간에는 배치별로 로드하는 걸 구성해보자.