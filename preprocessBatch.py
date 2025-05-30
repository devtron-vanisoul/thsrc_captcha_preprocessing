#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, random, cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import binarize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

WIDTH = 140
HEIGHT = 48

CAPTCHA_FOLDER = "captcha/"
PROCESSED_FOLDER = "processed/"


# In[ ]:


def imgDenoise(filename):
    img = cv2.imread(filename)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 30, 30, 7, 21)
    return dst


# In[ ]:


def img2Gray(img):
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh


# In[ ]:


def findRegression(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[:, 14:WIDTH - 7] = 0
    imagedata = np.where(img == 255)

    X = np.array([imagedata[1]])
    Y = HEIGHT - imagedata[0]

    poly_reg = PolynomialFeatures(degree = 2)
    X_ = poly_reg.fit_transform(X.T)
    regr = LinearRegression()
    regr.fit(X_, Y)
    return regr


# In[ ]:


def dePolynomial(img, regr):
    X2 = np.array([[i for i in range(0, WIDTH)]])
    poly_reg = PolynomialFeatures(degree = 2)
    X2_ = poly_reg.fit_transform(X2.T)
    offset = 4

    newimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for ele in np.column_stack([regr.predict(X2_).round(2), X2[0]]):
        pos = HEIGHT - int(ele[0])
        newimg[pos - offset:pos + offset, int(ele[1])] = 255 - newimg[pos - offset:pos + offset, int(ele[1])]

    return newimg


# In[ ]:


def addPadding(img):
    size = (WIDTH - HEIGHT) // 2
    const = cv2.copyMakeBorder(img, size, size, 0, 0, cv2.BORDER_CONSTANT, value = [0,0,0])
    return const


# In[ ]:


def enhanceText(img):
    """
    針對文字識別優化的圖片增強
    """
    # 1. 自適應直方圖均衡化
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # 2. 雙邊濾波保邊去噪
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # 3. 自適應閾值二值化
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # 4. 形態學操作清理噪點
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 5. 文字筆劃增強
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary = cv2.dilate(binary, kernel_dilate, iterations=1)

    return binary

def advancedTextEnhancement(img):
    """
    更進階的文字增強方法
    """
    # 1. Gamma校正增強對比度
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)

    # 2. 銳化濾波
    kernel_sharp = np.array([[-1,-1,-1],
                            [-1, 9,-1],
                            [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel_sharp)

    # 3. 多尺度Retinex增強
    img = multiScaleRetinex(img)

    # 4. 自適應二值化 (Otsu + Gaussian)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. 連通組件清理
    binary = cleanSmallComponents(binary)

    return binary

def multiScaleRetinex(img, scales=[15, 80, 250]):
    """
    多尺度Retinex演算法增強圖片
    """
    img = img.astype(np.float32) + 1.0
    retinex = np.zeros_like(img)

    for scale in scales:
        gaussian = cv2.GaussianBlur(img, (0, 0), scale)
        retinex += np.log10(img) - np.log10(gaussian)

    retinex = retinex / len(scales)
    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex)) * 255
    return retinex.astype(np.uint8)

def cleanSmallComponents(binary, min_size=50):
    """
    清理小的連通組件
    """
    # 找到連通組件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 創建清理後的圖片
    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels):  # 跳過背景(標籤0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255

    return cleaned

def textSpecificBinarization(img):
    """
    專門針對文字的二值化方法
    """
    # 1. 預處理
    img = cv2.medianBlur(img, 3)

    # 2. 嘗試多種閾值方法並選擇最佳
    methods = []

    # Otsu
    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    methods.append(otsu)

    # 自適應閾值 (均值)
    adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 15, 8)
    methods.append(adaptive_mean)

    # 自適應閾值 (高斯)
    adaptive_gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 15, 8)
    methods.append(adaptive_gaussian)

    # 選擇文字區域最清晰的方法
    best_method = selectBestBinarization(methods, img)

    return best_method

def selectBestBinarization(methods, original):
    """
    基於文字清晰度選擇最佳二值化方法
    """
    scores = []

    for method in methods:
        # 計算邊緣強度作為清晰度指標
        edges = cv2.Canny(method, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # 計算連通組件數量 (適中的組件數量通常對應好的文字分割)
        num_labels, _ = cv2.connectedComponents(method)
        component_score = 1.0 / (1.0 + abs(num_labels - 10))  # 假設理想組件數為10

        # 綜合評分
        score = edge_density * 0.7 + component_score * 0.3
        scores.append(score)

    best_idx = np.argmax(scores)
    return methods[best_idx]


# In[ ]:


dic = [[0] * 2 for i in range(100)]
for i in range(100):
    dic[i][0]=25
    dic[i][1]=25
dic[50][0]=26
dic[50][1]=24
dic[48][0]=23
dic[48][1]=30
dic[46][0]=27
dic[46][1]=25
dic[45][0]=21
dic[45][1]=30

def preprocessing(from_filename, to_filename):
    # 載入影像
    img = cv2.imread(from_filename)
    if img is None:
        raise FileNotFoundError(f"找不到檔案: {from_filename}")

    height1, width1, _ = img.shape

    # 降噪
    dst = cv2.fastNlMeansDenoisingColored(img, None, 31, 31, 7, 21)

    # 儲存降噪圖片到暫存（避免白邊）
    img_bgr = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)

    # 以 figsize=(width1, height1) 為基礎計算最終輸出解析度
    output_width = int(width1 * 10)   # dpi=10
    output_height = int(height1 * 10)

    # 重新調整圖像大小
    img_resized = cv2.resize(img_bgr, (output_width, output_height), interpolation=cv2.INTER_AREA)

    # 讀取降噪後圖片再黑白化
    ret, thresh = cv2.threshold(img_resized, 127, 255, cv2.THRESH_BINARY_INV)
    height, width, _ = thresh.shape
    imgarr = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    # 區域遮罩處理
    imgarr[:, 100:width-40] = 0
    imagedata = np.where(imgarr == 255)
    X = np.array([imagedata[1]])
    Y = height - imagedata[0]

    # 曲線擬合
    poly_reg = PolynomialFeatures(degree=2)
    X_ = poly_reg.fit_transform(X.T)
    regr = LinearRegression()
    regr.fit(X_, Y)

    # 回歸線預測
    X2 = np.array([[i for i in range(0, width)]])
    X2_ = poly_reg.fit_transform(X2.T)

    for ele in np.column_stack([regr.predict(X2_).round(0), X2[0]]):
        pos = height - int(ele[0])
        start_y = max(pos - int(dic[height1][0]), 0)
        end_y = min(pos + int(dic[height1][1]), height)
        thresh[start_y:end_y, int(ele[1])] = 255 - thresh[start_y:end_y, int(ele[1])]

    # 調整大小並儲存
    thresh = 255 - thresh
    newdst = cv2.resize(thresh, (140, 48), interpolation=cv2.INTER_AREA)
    if newdst.dtype != 'uint8':
        newdst = (newdst * 255).astype('uint8')  # 若原圖是 0~1 浮點數

    # 儲存圖片
    cv2.imwrite(to_filename, newdst)


# In[ ]:


i = 0

# ignore existing image
while True:
    i += 1
    filename = PROCESSED_FOLDER + str(i) + '.jpg'
    if not os.path.isfile(filename):
        i -= 1
        break

print("start to process image from index: " + str(i + 1))

while True:
    i += 1
    filename = CAPTCHA_FOLDER + str(i) + '.jpg'
    if not os.path.isfile(filename):
        break
    preprocessing(filename, PROCESSED_FOLDER + str(i) + '.jpg')
    print("i: ", i)

print("completed")


# In[ ]:




