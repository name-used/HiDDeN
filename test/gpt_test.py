import cv2
import numpy as np
import jassor.utils as J

# 1. 读取彩色图像
image = cv2.imread('cat.jpg')  # 将 'input.jpg' 替换为你的图片路径
if image is None:
    print("无法读取图片，请检查路径")
    exit()

# 2. 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. 反转灰度图像的像素值
inverted_gray = 255 - gray_image

# 4. 使用高斯模糊
blurred = cv2.GaussianBlur(inverted_gray, (21, 21), sigmaX=0, sigmaY=0)

# 5. 反转模糊图像
inverted_blur = 255 - blurred

# 6. 创建素描效果
sketch = cv2.divide(gray_image, inverted_blur, scale=256.0)

# 7. 显示结果
# cv2.imshow('Original Image', image)
# cv2.imshow('Sketch', sketch)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 保存结果
# cv2.imwrite('cat.jpg', sketch)
J.plots([image, np.stack([sketch]*3, axis=2), gray_image > inverted_blur])
