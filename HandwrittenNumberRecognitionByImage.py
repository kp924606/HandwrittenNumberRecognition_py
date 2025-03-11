#The following code is based on the "AI Application Development Course" and has been partially modified after further personal research.

import cv2
import keras
import numpy as np
import threading

# 定義顯示影像的函式
def show_img():    
    window_name = 'Resized Image'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 允許調整視窗大小
    cv2.resizeWindow(window_name, 300, 300)  # 設定視窗大小 (寬, 高)
    cv2.moveWindow(window_name, 100, 100)  # 移動視窗到 (X, Y) 位置
    cv2.imshow(window_name, img)  # 顯示調整後的圖片
    cv2.waitKey(0)  # 等待使用者按下任意鍵
    cv2.destroyAllWindows()  # 關閉視窗    

#匯入模型(MNIST), 可使用"HandwrittenNumberRecognitionTrain.py"來訓練模型
model=keras.models.load_model('mnist.keras')
#匯入指定的圖片
img = cv2.imread('test1.jpg')
#把圖片尺寸縮到28x28
img = cv2.resize(img,(28,28))

# 啟動獨立執行緒顯示圖片（非同步）
threading.Thread(target=show_img, daemon=True).start()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #改變成黑白

    # pixel=[]
    # for i in range(28):
    #     for j in range(28):
    #         pixel.append((255-gray[i,j])/255)
    # pixelarray=np.asarray([pixel])

#符合 MNIST 手寫數字模型的輸入需求.MNIST 模型的輸入通常是 0~1 之間的浮點數
#反轉顏色,這個計算會將黑色變成白色，白色變成黑色,將數據標準化到 0~1
pixelarray = np.array([(255 - gray) / 255]).reshape(1, -1)

label=model.predict(pixelarray) #predict=辨識影像並取得結果
maxindex = np.argmax(label)
if label.max()>0.6: #限制機率要超過60%
    print("辨識結果:",maxindex)
else:
    print('無法確認')

# 等待使用者按下任何按鍵+Enter退出
input("按下任何按鍵+Enter退出...")

