#The following code is based on the "AI Application Development Course" and has been partially modified after further personal research.

#安裝pip install opencv-contrib-python
import tkinter as tk #Py圖形介面
#import tk
from PIL import ImageGrab
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
import numpy as np
import os
import threading
import ctypes
#啟用 tkinter 的 DPI Awareness, 原本作業系統視窗比例問題(Win10以上會預設125%), 用這招可以固定100%不被縮放影響.
ctypes.windll.shcore.SetProcessDpiAwareness(1)  # 設定為 System DPI Aware

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model=tf.keras.models.load_model('mnist.keras') #匯入之前訓練的模型存檔的位置,傳統神經網路
#model=tf.keras.models.load_model('CNNMnist.keras') #之前記得模型存檔的位置,CNN 捲基

fileName="testByManual1.jpg" #未來圖片存檔
#設定小畫家視窗
width = 280
height = 280
white = (255, 255, 255)

# 儲存目前的圖像視窗句柄
image_window = None  # 初始化為 None，避免出現 NameError

#滑鼠畫圖
def paint(event):
    x1, y1 = (event.x + 1), (event.y + 1)
    x2, y2 = (event.x - 1), (event.y - 1)
    canvas1.create_oval(x1, y1, x2, y2, fill="black", width=15)  # On tkinter Canvas

#清除畫面
def clear ():
    # Clear the SEEN canvas
    canvas1.delete('all')    
    close_image_window()  # 清除按鈕時關閉圖像視窗

# 定義關閉圖像視窗的函式
def close_image_window():
    global image_window
    if image_window is not None:
        plt.close()  # 關閉圖像視窗
        image_window = None

#顯示繪製的圖案
def show_image(pixel):    
    global image_window    
    if image_window is not None:
        image_window.set_data(pixel)  # 更新圖片數據
        plt.draw()  # 刷新圖像
    else :
        fig = plt.gcf()
        fig.set_size_inches(2, 2)
        image_window = plt.imshow(pixel, cmap='binary')  # cmap='binary' 參數設定以黑白灰階顯示    
        plt.show()  # 顯示圖像

#存檔
def predict():
    #存檔
    x=root.winfo_rootx()
    y=root.winfo_rooty()
    x1=x+canvas1.winfo_width()
    y1=y+canvas1.winfo_height()
    ImageGrab.grab().crop((x,y,x1,y1)).save(fileName)
    #辨識
    img = cv2.resize(cv2.imread(fileName),(28,28))
    pixel = (255-cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) /255 # RGB->GRAY

    show_image(pixel)
    # 啟動新執行緒顯示圖片（非同步）
    #threading.Thread(target=show_image, args=(pixel,)).start()

        # fig = plt.gcf()  
        # fig.set_size_inches(2,2)  
        # plt.imshow(pixel, cmap='binary') # cmap='binary' 參數設定以黑白灰階顯示.    
        # plt.show() #預覽寫字的結果
     
    pixelarray=pixel.reshape(-1,28,28,1) #轉成輸入陣列<-CNN， -1-->自動分配
    pixelarray=np.asarray([pixel]) #轉成輸入陣列<-傳統類神經
    pixelarray=pixelarray.reshape(1,-1) #全部保留,拉平(傳統神經網路)
    label=model.predict(pixelarray) # 用MNIST模型來預測
    maxindex = np.argmax(label)#找出0,1,2,...9，機率最大的輸出
    #print(label) #顯示機率矩陣
    if label.max()>0.5:
        showtext="辨識結果=" + str(maxindex) + ", 機率=" + str(label[0][maxindex])
        print(showtext)        
        textValue.set(showtext)
    else:
        print("無法辨識")

# 建立GUI視窗
root = tk.Tk()
root.tk.call('tk', 'scaling', 1.5)  # 設定按鈕縮放比例(原本1會太小)

# 建立畫布cv
canvas1 = tk.Canvas(root, width=width, height=height, bg='white')
canvas1.pack()
canvas1.bind("<B1-Motion>", paint) #設定滑鼠按下為繪圖

#建立辨識按鈕及清除按鈕
textValue = tk.StringVar()
textValue.set('')
label1=tk.Label(textvariable=textValue).pack()
button=tk.Button(text="辨識", command=predict).pack()
button=tk.Button(text="清除", command=clear).pack()

root.mainloop()
