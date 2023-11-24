---
title: dotshow
layout: default
parent: 자체 개발 라이브러리
nav_order: 1
---

# dotshow
    We generate image to text on terminal
    It is made for terminal (CLI) to show image.
    When we want to see image on python  linux server, 
    we have to save image or use jupyter notebook.
    To overcome those problem, I made this package.

    To see thumnail of the image, we can use cv2.imshow in local environment.
    However, it is impossible for server ssh terminal.
    It is simple thumnail printer using python.
    

# Example
    We can print images like down below.
### 원본 이미지
<img width="300" alt="before" src="https://user-images.githubusercontent.com/50725139/140743113-9db67704-0a93-4f58-9542-a893b915a543.png">

### dotshow
<img width="300" alt="dotshow" src="https://user-images.githubusercontent.com/50725139/140743199-64cac4d2-08be-4b23-9f21-393b2577bc51.png">

### colorshow
<img width="300" alt="colorshow" src="https://user-images.githubusercontent.com/50725139/201597516-fd367a3b-5106-4b75-b02f-4435a00f54b9.png">


<!-- <img width="300" align = "left"  alt="before" src="https://user-images.githubusercontent.com/50725139/140743399-5daf658c-085e-44f5-8e65-d9821f53512d.png">
<img width="300" alt="after" src="https://user-images.githubusercontent.com/50725139/140743425-35af69bf-3aca-4105-9c3b-4540b846ad7f.png"> -->


# How to use
### Git cloning on your repository
    pip install dotshow
    
## import package on python
### Draw image on terminal by putting path
```python
from dotshow import loadshow
loadshow(<img-path>) # run the code (drawing a color image, default=color)
loadshow(<img-path>, color=False) # run the code (drawing a gray image)
```

### Draw image on terminal by OpenCV2 array
```python
import cv2
from dotshow import dotshow, colorshow
img = cv2.imread(<img-path>)
dotshow(img) # run the code (drawing a gray image)
colorshow(img) # run the code (drawing a color image)
````

### Draw image on terminal by PIL Image
```python
import numpy as np
from PIL import Image
from dotshow import dotshow, colorshow
img = np.array(Image.open(<img-path>))
dotshow(img) # run the code (drawing a gray image)
colorshow(img) # run the code (drawing a color image)
```

# Parameters
    loadshow(
        gray = True[default / bool]
        size = 7 [default / 0 ~ 10]
    )
    dotshow(
        gray = True[default]
        size = 7 [default / 0 ~ 10]
    )
    colorshow(
        size = 7 [default / 0 ~ 10]
    )
    
    gray - if your image is not interpretable, give gray parater False
    size - if your image is too big in your terminal, reduce size using this parameter
    
* If you use low size parameter, image will undersampled too much and it will decrease image quality.
* If you use big size parameter, the terminal cannot describe image because the terminal is too small.

    
