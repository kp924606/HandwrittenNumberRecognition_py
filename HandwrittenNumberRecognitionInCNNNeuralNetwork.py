#The following code is based on the "AI Application Development Course" and has been partially modified after further personal research.

import matplotlib.pyplot as plt #繪圖工具
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout #Conv2D>卷積神經網路,提取影像中的特徵 (如邊緣、紋理、形狀等).Dropout>刻意忘記. MaxPooling2D 池化>只選最大值. Flatten>將多維資料攤平成一維.
#import pandas as pd #pip install pandas
from keras.datasets import mnist

from tensorflow.keras.utils import to_categorical
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#1.資料收集, 會自動下載並載入 MNIST 數據, 包含了 60,000 張訓練圖片和 10,000 張測試圖片
#X_train、X_test 是影像資料（28×28 的數字圖片）
#Y_train、Y_test 是對應的數字標籤（0~9）
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=x_train.reshape((60000, 28, 28, 1))/255.0 # 1>影像的通道數 (channels),數據集的圖片是 灰階影像，每個像素只有一個亮度值. /255>歸1化
x_test=x_test.reshape((10000, 28, 28, 1))/255.0
y_train = to_categorical(y_train, num_classes=10) #[3]->[0,0,0,1,0,0,0,0,0,0,0]
y_test = to_categorical(y_test, num_classes=10)

def plot_image(image):  
    fig = plt.gcf()  
    fig.set_size_inches(2,2)  
    plt.imshow(image, cmap='binary') # cmap='binary' 參數設定以黑白灰階顯示.  
    plt.show()
plot_image(x_train[0])

#2.建立Sequential模型
model = tf.keras.models.Sequential()
#每次卷積後，圖形大小會減少卷積-1,每次池化大小會/2
#16個3x3卷積核                   數量:8的倍數   大小:3,5,7奇數         input_shape:長28x寬28x顏色1
#16x26x26
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
#池化>16x13x13
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2)) #隨機遺忘20%
#32個5x5卷積核
#32x11x11
model.add(Conv2D(32, (3, 3), activation='relu'))
#池化>32x5x5
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.2))
#轉平面層
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax')) #softmax>機率值 最後一層, 輸出轉換為一組概率值,使得這些概率值的總和為 1.units=10 代表輸出有0~9共10種可能.
model.summary() # 顯示模型結果
model.compile(optimizer='adam', #tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#訓練模型                                  48000/200=2400,  2400*50=120000 verbose:一輪回報一次
train_history =model.fit(x_train, y_train,batch_size=200, epochs=20,validation_split=0.2, verbose=1)
#批次量:每次完成1個批次(對答案),就更新神經網路
#60000x80%=48000(訓練用)
#1輪240批(48000/200)

#顯示訓練過程
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(train_history, 'accuracy', 'val_accuracy')

#4.評估模型
loss, accuracy = model.evaluate(x_test, y_test)
print('test loss: ', loss)
print('test accuracy: ', accuracy)
#print(pd.crosstab(y_test, prediction,rownames=['實際'], colnames=['預測']))

#5.儲存模型檔案
model.save('CNNMnist.keras')

