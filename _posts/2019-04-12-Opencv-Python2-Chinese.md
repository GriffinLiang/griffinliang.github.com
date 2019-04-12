<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# OpenCV和Python2中的中文问题

---

## 中文字体安装（Ubuntu系统）
 * 下载 simsun.ttc 字体，复制到/usr/share/fonts/truetype/simsun/路径下。

## 代码示例
 * 系统采用uft-8编码
 * 中文解码使用gbk
 * 可视化需要使用PIL进行中转，借鉴网上的教程。
 
```python
# coding:utf-8
import sys
# sys.setdefaultencoding() does not exist, here!
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('utf-8')
print sys.getdefaultencoding()

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

#coding=utf-8
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        print img.dtype, img.shape
        img = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("font/truetype/freefont/simsun/simsun.ttc", 20, encoding="utf-8")
    draw.text((left, top), text.decode('gbk'), textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

vis_img = np.zeros((512,512,3), np.uint8)
vis_img = cv2ImgAddText(vis_img, '中文', 200, 300, (255, 255, 255), 20)
cv2.imwrite('test.png', vis_img.astype(np.uint8))
```