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


def preprocessing(from_filename, to_filename):
    img = cv2.imread(from_filename)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret1,img_ = cv2.threshold(img,1234, 255,cv2.THRESH_OTSU)
    ret,img = cv2.threshold(img,ret1+5, 255,cv2.THRESH_BINARY)

    img = cv2.fastNlMeansDenoising(img, None, 50.0, 7, 50)
    ret1,img = cv2.threshold(img,1234, 255,cv2.THRESH_OTSU)

    def find_point(mat):
        start_index=0
        end_index=mat.shape[0]-1
        for i in range(mat.shape[0]):
            if np.all(mat[i]==0) and start_index==0:
                start_index=i
            elif np.all(mat[i]==0)==False and start_index!=0:
                end_index=i-1
                break
        return int(start_index+round((end_index-start_index+1)/2))

    left=img[:, :5]
    right=img[:, -5:]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    left=cv2.erode(left, kernel)
    right=cv2.erode(right, kernel)
    ret, left=cv2.threshold(left,1234, 255,cv2.THRESH_OTSU)
    ret, right=cv2.threshold(right,1234, 255,cv2.THRESH_OTSU)

    left_point=find_point(left)
    right_point=find_point(right)
    new_x=[i for i in range(img.shape[1])]
    new_y=np.poly1d(np.polyfit([0, (img.shape[1]-1)/2, img.shape[1]-1], [left_point, right_point+((left_point-right_point)/2)-6, right_point], 2))(new_x)
    line_template=np.full(img.shape, 255, dtype=np.uint8)
    for x_, y_ in zip(new_x, new_y):
        y_=int(round(y_))
        color=255-img[y_][x_]
        if color==255:
            cv2.line(line_template, (x_, y_), (x_, y_), (0, 0, 0), 3, cv2.LINE_AA)
        elif color==0:
            cv2.line(line_template, (x_, y_), (x_, y_), (0, 0, 0), 3, cv2.LINE_AA)
    line_template=line_template.astype(np.uint8)

    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if line_template[r][c]==0 and img[r][c]==0:
                img[r][c]=255
            elif line_template[r][c]==0 and img[r][c]==255:
                img[r][c]=0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 3))
    img=cv2.erode(img, kernel)
    img=cv2.dilate(img, kernel)
    img=cv2.dilate(img, kernel)
    img=cv2.erode(img, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 1))
    img=cv2.dilate(img, kernel)
    img=cv2.erode(img, kernel)

    h=200
    w=int(img.shape[1]*h/img.shape[0])
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    bordersize = 100
    img = cv2.copyMakeBorder(
        img,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    # 上下切 80, 左右切 100
    img = img[80:img.shape[0]-80, 100:img.shape[1]-100]
    # resize to 140 48
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(to_filename, img)
    return

