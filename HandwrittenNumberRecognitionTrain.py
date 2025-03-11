#The following code is based on the "AI Application Development Course" and has been partially modified after further personal research.

#import tensorflow as tf
import keras as ks
#print(tf.__version__)
print(ks.__version__)
import matplotlib.pyplot as plt
from keras.models import Sequential #順序的神經網路, 用來構建 線性堆疊（stack） 的神經網絡。這模型的每一層都是依照順序堆疊的,並且每一層的輸出會成為下一層的輸入.這種結構非常適合於層與層之間有順序關係的情況.可以將不同的層（例如 Dense、Dropout 等）添加到這個模型中.
from keras.layers import Dense, Dropout #Dense一層神經網路, Dropout 遺忘,隨機“遺忘”神經網絡中的一部分神經元,把某些神經元的輸出設為 0,可以強迫模型不依賴特定的神經元,進而提升模型的泛化能力.
#MNIST 包含 70,000 張 28×28 像素的手寫數字圖片，這些圖片來自美國高中生及美國人口普查局工作人員的手寫數字。
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical #[4]=[0,0,0,0,1,0,0,0,0,0] 機率陣列
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#1.資料收集, 會自動下載並載入 MNIST 數據, 包含了 60,000 張訓練圖片和 10,000 張測試圖片
#X_train、X_test 是影像資料（28×28 的數字圖片）
#Y_train、Y_test 是對應的數字標籤（0~9）
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

def plot_image(image):  
    fig = plt.gcf()  
    fig.set_size_inches(3,3)  
    plt.imshow(image, cmap='binary') # cmap='binary'  
    plt.show()
#顯示數字圖案, 記得要關閉視窗才會往下繼續執行
#plot_image(X_train[0])

# 配合神經網路結構  reshape>陣列重新分配      -1>自動計算 /255>歸1化
X_train = X_train.reshape(X_train.shape[0], -1)/255 #將 28×28 矩陣攤平成 784 維的向量，讓 MLP（全連接神經網路）可以處理.歸一化（Normalization）：:將像素值從 0~255 轉成 0~1，
X_test = X_test.reshape(X_test.shape[0], -1)/255
Y_train = to_categorical(Y_train, num_classes=10) #[3]->[0,0,0,1,0,0,0,0,0,0,0].原本 Y_train 的數字標籤是 0~9，但神經網路需要機率分佈的輸出,所以轉換成 One-Hot 編碼.
Y_test = to_categorical(Y_test, num_classes=10)

#2. 建立模型
model = Sequential()
#在模型中添加一個神經網路,它包含 256 個神經元, 輸入維度是 784（784 維的扁平化圖片,像是 28x28 像素的手寫數字圖片）
model.add(Dense(units=256, input_dim=784,activation='relu')) #relu 激發函數, 對於輸入的每一個值,ReLU 都會將負數變為 0,正數保持不變.
model.add(Dropout(0.2)) #隨機遺忘20%
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.2)) #隨機遺忘20%
model.add(Dense(units=64,activation='relu'))
model.add(Dropout(0.2)) #隨機遺忘20%
model.add(Dense(units=32,activation='relu'))
model.add(Dropout(0.2)) #隨機遺忘20%
model.add(Dense(units=10,activation='softmax')) #softmax>機率值 最後一層, 輸出轉換為一組概率值,使得這些概率值的總和為 1.units=10 代表輸出有0~9共10種可能.
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary() #顯示神經網絡模型的簡要總結

#3.模型訓練
#                    訓練 X自變數     Y應變數     20%做測試valida, 剩下的80%做訓練    epochs=模型在訓練過程中看到每一個樣本一次後    訓練次數=批次量:一批200個 verbose=回報次數:1輪1次
train_history = model.fit(x=X_train, y=Y_train , validation_split=0.2, epochs=30, batch_size=200, verbose=1)  
#批次量:每次完成1個批次(對答案),就更新神經網路
#60000x80%=48000(訓練用)
#1輪240批(48000/200)

#poch 20/20
#240/240 [==============================] - 0s 2ms/step - loss: 0.0138 - accuracy: 0.9975 - val_loss: 0.0852 - val_accuracy: 0.9755
#loss 和 accuracy 是背題庫的結果. val_loss 和 val_accuracy 是考試結果.

#4 評估模型
loss, accuracy = model.evaluate(X_test, Y_test)
print('test loss: ', loss)
print('test accuracy: ', accuracy)

# 顯示訓練過程中的歷史數據
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    plt.show()
show_train_history(train_history, 'accuracy', 'val_accuracy')  


#5. 將模型儲存
model.save('mnist.keras')


