---
title: L6 Overloaded
layout: home
---

* 구조체에서 함수에 입력을 주기 위하 다양한 방법
* Mojo에서는 함수의 Type을 지정하면 이를 변경할 수 없기 때문에 다양한 Type이나 입력 개수 등에 따라 입력을 다르게 처리할 수 있도록 한다.

* 서로 다른 방식으로 초기화를 해줄 수 있는데, 타입과 매게변수에 따라 다른 함수가 호출된다. 이를 **Overload**라고 한다.

```python
struct MyFloat:
	var _int: Int
	var _decimal: Float32
  
	# 1. 매게션수가 하나이면서 타입이 Int인 경우
	fn __init__(inout self, x: Int):
		self._int = x
		self._decimal = 0.0

	# 2. 매게변수가 하나이면서 타입이 Float32인 경우
	fn __init__(inout self, d: Float32):
		self._int = 0
		self._decimal = d

	# 3. 매게변수가 두개이면서 타입이 Int, Float32인 경우
	fn __init__(inout self, x: Int, d: Float32):
		self._int = x
		self._decimal = d

	fn get_output(self) -> Float32:
		return self._int + self._decimal
```

```python
let x :Int = 10
let d :Float32 = 0.5
let myFloat1 = MyFloat(x)
print(myFloat.get_output()) # 10.0

let myFloat2 = MyFloat(d)
print(myFloat.get_output()) # 0.5

let myFloat3 = MyFloat(x, d)
print(myFloat2.get_output()) # 10.5
```

* 이런식으로 다양한 입력 방식에 따라 다르게 설정할 수 있다.
