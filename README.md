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

### 圖表：
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

## 1-5. MNIST（Modified National Institute of Standards and Technology）數據集
MNIST 是一個經典的手寫數字資料集，主要用於機器學習和深度學習的圖像分類任務。這個數據集由美國國家標準與技術研究院（NIST）整理，並經過修改以適應機器學習模型的訓練需求。

### 主要功能：
- 神經網路訓練：如 MLP（多層感知機）、CNN（卷積神經網路） 等。
- 影像識別測試：評估不同演算法的效能，如 KNN、SVM、決策樹等。
- 模型比較：研究不同 AI 模型的準確度，如 TensorFlow、PyTorch。
- 手寫識別應用：如數位簽名驗證、銀行票據識別等。

### 特點：
- 1.數據內容：
 
  包含 70,000 張手寫數字圖片：
  
    訓練集：60,000 張
  
    測試集：10,000 張
  
  每張圖片都是 28×28 像素，灰階圖像（0~255 像素值）。
  
  標籤為 0 到 9（代表手寫數字 0~9）。

- 2.格式：
  
  影像（X）：大小為 (28,28) 的 NumPy 陣列。
  
  標籤（Y）：數字 0~9，通常會轉成 One-Hot 編碼（如 4 → [0,0,0,0,1,0,0,0,0,0]）。

數字5

![image](https://github.com/user-attachments/assets/4a11e03c-6420-4a06-b6c3-c6b0282e2ddb)

------

## 2. Install Command：
Please refer the command as below.

## 2-1. matplotlib 
```bash
pip install matplotlib
```

## 2-1. tensorflow & keras

if you have nVidia GPU.
```bash
conda install tensorflow-gpu==2.6
conda install keras==2.6
pip install h5py
pip install numpy==1.23.4
pip install opencv-contrib-python
```

if you haven't nVidia GPU.
```bash
conda install tensorflow
conda install keras==2.6
pip install h5py
pip install numpy==1.23.4
pip install opencv-contrib-python
```

------

## 3. py Code：

## 3-1. HandwrittenNumberRecognitionTrain.py
使用 Keras 建立了一個 基於 MLP 的深度學習模型,來辨識手寫數字,透過 MNIST 數據集訓練、驗證、測試、並儲存模型,最終能夠準確識別 0~9 的手寫數字。

### 主要功能：

- 1.資料收集與前處理

  下載 MNIST 數據集（包含 60,000 張訓練圖片與 10,000 張測試圖片）。
  
  影像數據轉換為 28×28 的矩陣並攤平成 784 維向量，進行歸一化（將像素值從 0255 轉為 01）。
  
  標籤（0~9 數字）轉換為 One-Hot 編碼，以便模型輸出機率分佈。

- 2.建立模型（Sequential 模型）

  採用 多層感知機（MLP），包含 5 層 Dense（全連接層）。
  
  每層使用 ReLU 激活函數，最後一層使用 Softmax 激活函數（轉換成 0~9 的機率分布）。
  
  加入 Dropout（隨機遺忘），防止過擬合。
  
  使用 Adam 優化器，損失函數為 categorical_crossentropy（交叉熵），衡量分類誤差。

- 3.訓練模型

  以 80% 訓練數據 訓練模型，20% 驗證數據 進行驗證。
  
  設定 30 個 epochs（訓練週期），每次訓練批次大小為 200。
  
  顯示訓練過程，監控 loss（損失值）與 accuracy（準確率）。

- 4.評估模型

  使用測試數據集評估模型表現，輸出測試損失（test loss）與測試準確率（test accuracy）。
  
  繪製訓練與驗證的 accuracy 變化趨勢圖，以觀察模型的學習過程。

- 5.儲存模型

  訓練完成後，將模型存為 mnist.keras，以便後續載入使用。

![image](https://github.com/user-attachments/assets/ad3c2073-4433-4510-9380-7e3870b40155)

![image](https://github.com/user-attachments/assets/7dafe992-6e6b-4166-9751-abb27339f550)

![image](https://github.com/user-attachments/assets/e7c1adee-990d-46f6-b4cc-ce5adf0ab261)

![image](https://github.com/user-attachments/assets/6d4e9fe2-74a8-4029-829c-b90757bcaf73)

![image](https://github.com/user-attachments/assets/e10c38ae-5db6-477c-aa43-eab7307b93fe)

------

## 3-2. HandwrittenNumberRecognitionByImage.py
使用 Keras 訓練好的 MNIST 手寫數字辨識模型(from HandwrittenNumberRecognitionTrain.py)，對指定圖片進行預測並輸出結果。

### 主要功能：
- 1.匯入必要的函式庫
  
  cv2（OpenCV）：用於讀取與處理圖片、顯示影像。
  
  keras：載入 已訓練的手寫數字辨識模型（mnist.keras）。
  
  numpy：進行數據處理與格式轉換。
  
  threading：使用多執行緒顯示影像，避免影響主程式執行。

- 2.讀取與處理影像
  
  載入模型（mnist.keras）：這是一個 預訓練的 MNIST 手寫數字辨識模型。
  
  載入圖片（test1.jpg）：讀取使用者提供的圖片作為識別對象。
  
  調整大小（cv2.resize(img, (28,28))）：將圖片縮放成 28x28 像素（符合 MNIST 模型的輸入格式）。
  
  顯示影像（使用多執行緒 show_img()）：非同步顯示調整後的影像，避免阻塞主程式。

- 3.影像前處理
  
  轉換為灰階（cv2.COLOR_BGR2GRAY）：將彩色圖片轉為 單色（黑白），因為 MNIST 模型只接受單通道輸入。
  
  反轉顏色並標準化：
  
  255 - gray：MNIST 訓練數據的手寫數字是白底黑字，而某些影像可能是黑底白字，因此需要進行 顏色反轉。
  
  除以 255：將數據歸一化，使像素值介於 0~1（這符合 MNIST 模型的輸入需求）。
  
  轉換為 1 維陣列 reshape(1, -1)，以符合模型輸入格式。

- 4.進行數字辨識
  
  模型預測（model.predict(pixelarray)）：輸入處理後的影像，讓模型進行 手寫數字分類預測，返回 10 個數字（0~9）的機率分布。
  
  找出機率最高的數字（np.argmax(label)）：取最大機率值對應的索引，即預測結果。
  
  設定辨識門檻（60%）：
  
  若最大機率值 > 0.6，則顯示辨識結果。
  
  否則，顯示「無法確認」。

- 5.程式結束
  
  等待使用者輸入後退出程式（input("按下任何按鍵+Enter退出...")）。

![image](https://github.com/user-attachments/assets/26ef1799-b91b-4efd-9d14-639f1e424360)

![image](https://github.com/user-attachments/assets/b538fa9b-5c98-40fa-96a5-2ee483ef5f3c)

![image](https://github.com/user-attachments/assets/17b18daf-e3a0-491c-abad-c7127cb744c7)

------

## 3-3. HandwrittenNumberRecognitionByManual.py
透過手寫輸入介面，使用 Keras 訓練好的 MNIST 手寫數字辨識模型，對使用者繪製的數字進行預測並顯示結果。

### 主要功能：
- 1.匯入必要的函式庫
  
  tkinter：建立 GUI 繪圖介面，讓使用者用滑鼠繪製手寫數字。
  
  PIL.ImageGrab：擷取畫布內容，將手寫數字存為圖片。
  
  cv2（OpenCV）：處理影像，包括 縮放、轉換灰階 等步驟，使其符合 MNIST 模型的輸入格式。
  
  matplotlib.pyplot：用於顯示影像，幫助使用者確認手寫輸入內容。
  
  tensorflow.keras：載入並使用 MNIST 預訓練模型 來辨識手寫數字。
  
  numpy：進行數據處理與格式轉換，使其符合模型輸入需求。
  
  os：設定環境變數，避免 TensorFlow 載入模型時發生錯誤。
  
- 2.GUI 介面與手寫輸入
  
  建立 Tkinter 視窗 (root)，並建立 畫布 (Canvas)，讓使用者用滑鼠繪製數字。

  paint(event)：當滑鼠 左鍵按住並移動 時，在畫布上畫出黑色筆跡。

  clear()：清空畫布，讓使用者重新繪製。

- 3.擷取手寫數字並進行辨識
  
  predict()：
  
    擷取畫布內容 (ImageGrab.grab())，將手寫內容存為圖片 (testByManual1.jpg)。
  
    影像前處理：
  
      縮放至 28x28 像素（cv2.resize()）。
      轉換為灰階（cv2.cvtColor()）。
      反轉顏色並標準化至 0~1（(255-gray)/255），符合 MNIST 模型格式。
  
    顯示影像：
  
      show_image(pixel)：以 Matplotlib 顯示處理後的 28x28 影像，讓使用者確認。

    辨識數字：
  
      model.predict(pixelarray)：將影像輸入 Keras MNIST 預訓練模型 進行數字預測。
      np.argmax(label)：找出 機率最高的數字。
      若 機率超過 50% (label.max() > 0.5)，則顯示辨識結果，否則顯示「無法辨識」。

- 4.顯示辨識結果
  
  使用 tk.Label 來動態顯示 辨識結果與機率。
  
  當按下「辨識」按鈕時：
  
      會 擷取畫布內容、處理圖片、預測數字並更新顯示文字。
  
  當按下「清除」按鈕時：
  
      會 清空畫布與影像視窗，讓使用者重新繪製。

請手動輸入數字
![image](https://github.com/user-attachments/assets/2b21f6c4-b3d5-44bc-986d-4b61410d241e)

記得按下Figure1的關閉按鈕,才會往下執行辨識功能.
![image](https://github.com/user-attachments/assets/9f76bce8-2d55-47f3-8a62-403123e174a9)

![image](https://github.com/user-attachments/assets/54b24c99-ebcb-4f53-9a50-d17fc1d58615)

![image](https://github.com/user-attachments/assets/b33ec720-2210-4481-b999-9939d3d4e30b)

------

## 3-4. HandwrittenNumberRecognitionInCNNNeuralNetwork.py
訓練 CNN 捲基模型來識別 MNIST 數據集中的手寫數字，並對模型進行評估與儲存。

### 主要功能：
- 1.資料載入與預處理：

  程式從keras.datasets.mnist中載入MNIST數據集，這是一個包含60,000張訓練圖片和10,000張測試圖片的數據集，每張圖片的大小是28x28像素，並且是灰階（單一顏色通道）。
  
  接著對影像進行數值歸一化處理（除以255），將每個像素值轉換到[0, 1]範圍內，並且將標籤轉換為one-hot編碼。
  
- 2.模型建立：

  使用Sequential模型來定義神經網路結構。

  第一層為一個卷積層（Conv2D），具有16個3x3的卷積核，激活函數使用ReLU，並指定輸入形狀為28x28x1。

  接著加入池化層（MaxPooling2D）進行降維處理，並使用Dropout層來隨機丟棄部分神經元以防止過擬合。

  第二層為另一個卷積層，使用32個3x3的卷積核，再加上池化層和Dropout層。

  然後將資料攤平（Flatten），並經過幾層全連接層（Dense），每層後面都跟著Dropout層來減少過擬合。

  最後一層是輸出層，使用softmax激活函數，輸出10個神經元對應於0-9的數字。

- 3.模型編譯與訓練：

  編譯模型時選擇了adam優化器，並使用categorical_crossentropy作為損失函數，指標設為準確率。
  
  使用訓練資料進行模型訓練，並且將20%的訓練數據用作驗證集，進行20個epoch的訓練。

- 4.顯示訓練歷程：

  訓練過程中的準確率（訓練集和驗證集）會被繪製出來，幫助觀察模型的訓練效果。

- 5.模型評估：

  使用測試資料（x_test和y_test）評估模型的損失和準確率。

- 6.儲存模型：

  訓練完成後，將訓練好的模型儲存為CNNMnist.keras檔案，方便未來載入與使用。

模型 Summary
![image](https://github.com/user-attachments/assets/01517ddd-bf50-4951-831e-0849e71c3ed1)

![image](https://github.com/user-attachments/assets/ad108483-3e94-4910-a552-9bdfd2e06a86)

![image](https://github.com/user-attachments/assets/ffe2a3eb-71ea-4ea8-94b0-351ee2e0f4ac)

------

## 3-5. HandwrittenNumberRecognitionInCNNNeuralNetworkByManual.py

### 主要功能：


- 1.手繪數字：

  使用 tkinter 來建立一個畫布，使用者可以在畫布上繪製手寫數字，這透過 paint() 函式來實現，當滑鼠左鍵按下並移動時，會在畫布上繪製黑色線條。

- 2.圖片儲存與預處理：

  當使用者按下 "辨識" 按鈕時，會將畫布內容儲存為圖片（testByManual2.jpg），並對這張圖片進行預處理。
  
  預處理包括將圖片轉為灰階並反轉顏色，然後將圖像尺寸調整為28x28像素，這是MNIST數字識別模型所需的標準尺寸。

- 3.模型預測：

  預處理後的圖像被轉換為四維陣列，並輸入到事先訓練好的CNN模型（CNNMnist.keras），進行手寫數字的辨識。

  模型返回的是一個機率矩陣，程式會找出最大機率對應的數字，並顯示辨識結果與機率。

- 4.顯示圖像：

  在預測過程中，程式會將處理過的數字圖像顯示出來，以便使用者查看。

- 5.清除畫布：使用者可以按下 "清除" 按鈕來清空畫布，並重設任何顯示的辨識結果。

手動輸入

![image](https://github.com/user-attachments/assets/50df81ff-65a9-4967-a018-360a4360a83f)

記得按下Figure1的關閉按鈕,才會往下執行辨識功能.

![image](https://github.com/user-attachments/assets/0bea0bfe-57c5-4a3a-a1e3-d0058a929611)

辨識結果

![image](https://github.com/user-attachments/assets/14300ddc-4a57-4108-88ed-feab602bc8bc)

辨識結果

![image](https://github.com/user-attachments/assets/40c222bb-e4ff-4f23-af40-c7aee97bc112)

------

## About Me
Thanks & Best Regards !

蔡承廷

​Senior Engineer of Semiconductor Product/Testing & ​Automation

Email: ​​kp924606@gmail.com

LinkedIn:https://www.linkedin.comin/tsai-cheng-ting/
