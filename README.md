![](https://img.shields.io/badge/Creater-TCT-FFFF00) ![](https://img.shields.io/badge/development-python-006400) ![](https://img.shields.io/badge/Version-3.9.18-blue)

# HandwrittenNumberRecognition
HandwrittenNumberRecognition_py/手寫數字辨識

# 1. Package Introduce

## 1-1. OpenCV (cv2)
（Open Source Computer Vision Library）是一個開源的計算機視覺和機器學習軟件庫，旨在提供各種視覺任務的高效解決方案。它被廣泛應用於影像處理、物體檢測、影像分類、面部識別、計算機視覺等領域。

### 主要功能：

- 基本處理：旋轉、縮放、裁剪、平移、翻轉、顏色空間轉換（例如：BGR ↔ RGB、灰階、HSV 等）。

- 濾波：使用濾波器（如高斯濾波、邊緣檢測濾波等）來進行降噪或邊緣檢測。

- 圖像增強：調整對比度、亮度、色調，進行直方圖均衡化等。


### 物體檢測與追蹤：

- Haar 特徵分類器：用於面部識別、人臉檢測。

- HOG（Histogram of Oriented Gradients）：用於人類檢測。

- SSD（Single Shot Multibox Detector）/YOLO（You Only Look Once）：用於實時物體檢測。

- 物體追蹤：追蹤目標物體，例如使用 Meanshift 和 Camshift 進行物體跟踪。

### 特徵檢測與匹配：

- 角點檢測：如 Harris 角點檢測。
- SIFT/SURF：尺度不變特徵轉換，用於特徵檢測和匹配。
- ORB（Oriented FAST and Rotated BRIEF）：用於更快的特徵匹配。

- 機器學習：
支持向量機（SVM）：用於分類。
神經網絡：OpenCV 提供了深度學習的接口，可以使用預訓練的模型，如 Caffe、TensorFlow 等。

- 視頻處理：
讀取視頻：通過 cv2.VideoCapture 讀取攝像頭或視頻檔案。

  視頻錄製：用 cv2.VideoWriter 寫入視頻，支援多種格式（如 .avi、.mp4）。
  
  包括解析度調整、幀率修改、視頻剪輯等。

- 圖像變換：
透視變換：進行視角變換、圖像扭曲。

  圖像分割：將圖像劃分為多個區域。

- 計算機視覺任務：
文字識別（OCR）：透過 tesseract 等工具進行文本識別。

  手勢識別、人臉識別、姿態估計 等。

- 跨平台：
支援多種操作系統，如 Windows、Linux、macOS，並提供多種語言接口，最常用的是 C++ 和 Python，但也有 Java 和其他語言支持。

與硬體的兼容性好：支援多種影像擷取裝置，包括 WebCAM、USB 攝像頭、專業相機等，並能高效處理來自不同來源的視頻流。

易於集成與擴展：易於與其他機器學習庫（如 TensorFlow、PyTorch）集成，也能與硬體設備、網路接口等進行整合。

------

## 1-2. keras

Keras 是一個開源的深度學習框架，它提供了一個高層次的 API，使得建立和訓練神經網絡變得簡單易懂。Keras 支援多種底層深度學習引擎（如 TensorFlow、Theano 和 CNTK），並且現在 Keras 是 TensorFlow 的核心組件之一。

Keras 提供了簡單易用的 API，使得我們可以更輕鬆地定義和訓練深度神經網絡，像是全連接層（Dense Layer）、卷積層（Convolutional Layer）、池化層（Pooling Layer）等。

### 主要功能：
- 建立深度神經網絡模型：可以快速搭建神經網絡架構，從簡單的感知器到複雜的卷積神經網絡（CNN）和循環神經網絡（RNN）。
- 模型訓練與評估：提供簡單的訓練接口，可以輕鬆調整學習率、批次大小等超參數。
- 多種資料預處理與正規化功能：例如：標準化、正規化、資料增強等功能。

### 結構：
- Sequential 模型：用於逐層堆疊神經網絡。
- Functional API：用於建立更複雜的架構，如多輸入、多輸出的模型。
- 層（Layer）：神經網絡的基本組件，例如 Dense, Conv2D, LSTM 等。
- 優化器（Optimizer）：用於更新神經網絡的權重，常用的有 Adam、SGD 等。
- 損失函數（Loss Function）：用於衡量模型的預測與實際結果之間的誤差。

------

## 1-3. tensorflow

TensorFlow 是一個由 Google 開發的開源深度學習框架，旨在支援各種機器學習和深度學習的需求。TensorFlow 的核心是用來建立、訓練和部署機器學習模型的框架，特別是在大規模分佈式環境中非常有效。它支援多種平台，包括桌面、行動設備和雲端，並且能夠在 CPU、GPU 或 TPU 上運行。

### 主要功能：
- 構建深度學習模型：TensorFlow 提供了多種建立神經網絡的方式，從簡單的感知器到複雜的卷積神經網絡（CNN）和循環神經網絡（RNN）。
- 分佈式運算：TensorFlow 支援分佈式計算，可以在多台機器上訓練模型，這對於處理大規模資料集非常有用。
- 自動微分：TensorFlow 提供了強大的自動微分功能，能夠自動計算神經網絡中每個參數的梯度，並且進行反向傳播（backpropagation）。
- 跨平台部署：TensorFlow 不僅支援桌面環境，還能夠在移動設備（Android、iOS）和嵌入式設備上運行模型，還可以將模型部署到雲端。
- 支援多種硬體加速：TensorFlow 可以利用 CPU、GPU 或 TPU 來加速模型的訓練和推論。


------

## 1-4. matplotlib 
matplotlib 是一個廣泛使用的 Python 庫，用來生成各種類型的圖形，包括折線圖、散點圖、條形圖、直方圖等，廣泛應用於數據科學、機器學習、統計分析等領域。它是一個 2D 繪圖庫，但也可以用來創建 3D 圖表。

### 主要功能：
- 數據可視化：提供了豐富的 API 用來繪製不同類型的圖表。
- 高效圖表：支持高效生成各種類型的靜態圖像，並且能夠進行自訂化設計。
- 多種繪圖樣式：支援各種繪圖格式，如 PNG、PDF、SVG 等。
- 與數據分析工具集成：可以與 numpy、pandas 等數據處理工具無縫整合。
- 交互式繪圖：能夠與 Jupyter Notebook 等交互式環境結合，實現圖形交互。

###  圖表：
- 折線圖（Line plot）：顯示資料點之間的連接線。
- 散點圖（Scatter plot）：用來顯示數據點的分佈情況。
- 條形圖（Bar chart）：用來顯示每個類別的數據量，通常用於類別型資料。
- 直方圖（Histogram）：顯示資料的頻率分佈。
- 圓餅圖（Pie chart）：顯示資料在各個區塊之間的百分比。

### 常見的 pyplot 函數：
import matplotlib.pyplot as plt
- plt.plot()：畫出折線圖。
- plt.scatter()：畫出散點圖。
- plt.bar()：畫出條形圖。
- plt.hist()：畫出直方圖。
- plt.pie()：畫出圓餅圖。
- plt.xlabel() 和 plt.ylabel()：設置 x 軸和 y 軸的標籤。
- plt.title()：設置圖表的標題。
- plt.legend()：顯示圖例。
- plt.show()：顯示圖形。

------

## 2. Install Command：
Please refer the command as below.

## 2-1. matplotlib 
```bash
#pip install matplotlib
```

## 2-1. tensorflow & keras
```bash
conda install tensorflow-gpu==2.6
conda install keras==2.6
pip install h5py
pip install numpy==1.23.4
pip install opencv-contrib-python
```

------

## About Me
Thanks & Best Regards !

蔡承廷

​Senior Engineer of Semiconductor Product/Testing & ​Automation

Email: ​​kp924606@gmail.com

LinkedIn:https://www.linkedin.comin/tsai-cheng-ting/


