import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z = Z + float(b)
    return Z

def conv_forward(A_prev, W, b, hparameters):
    # Lấy kích thước từ A_prev
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Lấy kích thước từ W
    (f, f, n_C_prev, n_C) = W.shape

    # Lấy thông tin từ "hparameters"
    stride = hparameters['stride']
    pad = hparameters['pad']

    # Tính toán kích thước của output CONV
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    # Khởi tạo output volume Z bằng zeros
    Z = np.zeros((m, n_H, n_W, n_C))

    # Tạo A_prev_pad bằng cách thêm padding cho A_prev
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):  # vòng lặp qua batch các ví dụ huấn luyện
        a_prev_pad = A_prev_pad[i]  # Chọn kích hoạt đã được thêm padding của ví dụ huấn luyện thứ i
        for h in range(n_H):  # vòng lặp qua trục dọc của output volume
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(n_W):  # vòng lặp qua trục ngang của output volume
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(n_C):  # vòng lặp qua các kênh (= số lượng bộ lọc) của output volume
                    # Sử dụng các góc để định nghĩa lát cắt (3D) của a_prev_pad
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Thực hiện tích chập (convolve) lát cắt (3D) với bộ lọc W và bias b, để lấy một neuron đầu ra
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])
    ### END CODE HERE ###

    # Đảm bảo kích thước output là đúng
    assert(Z.shape == (m, n_H, n_W, n_C))

    # Lưu thông tin vào "cache" để sử dụng cho backpropagation
    cache = (A_prev, W, b, hparameters)

    return Z, cache

def pool_forward(A_prev, hparameters, mode = "max"):
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]

    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))

    ### START CODE HERE ###
    for i in range(m):  # loop over the training examples
        for h in range(n_H):  # loop on the vertical axis of the output volume
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(n_W):  # loop on the horizontal axis of the output volume
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(n_C):  # loop over the channels of the output volume

                    # Use the corners to define the current slice on the ith training example of A_prev, channel c
                    a_slice_prev = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    # Compute the pooling operation on the slice. Use an if statement to differentiate the modes.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_slice_prev)
    ### END CODE HERE ###

    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)

    # Making sure your output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))

    return A, cache

def conv_backward(dZ, cache):
    ### START CODE HERE ###
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache

    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):  # loop over the training examples

        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):  # loop over vertical axis of the output volume
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(n_W):  # loop over horizontal axis of the output volume
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(n_C):  # loop over the channels of the output volume

                    # Find the corners of the current "slice"
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        # Set the ith training example's dA_prev to the unpadded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        if pad != 0:
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad
    ### END CODE HERE ###

    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db

def create_mask_from_window(x):
    ### START CODE HERE ### (≈1 line)
    mask=x==np.max(x)
    ### END CODE HERE ###

    return mask

def distribute_value(dz, shape):
    ### START CODE HERE ###
    # Lấy kích thước từ shape
    (n_H, n_W) = shape

    # Tính toán giá trị để phân phối trên ma trận
    average_value = dz / (n_H * n_W)

    # Tạo một ma trận mà mỗi phần tử là giá trị "trung bình"
    a = np.full((n_H, n_W), average_value)
    ### END CODE HERE ###

    return a

def pool_backward(dA, cache, mode="max"):
    (A_prev, hparameters) = cache
    stride = hparameters["stride"]
    f = hparameters["f"]
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dA.shape
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask * dA[i, h, w, c]

                    elif mode == "average":
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)

    assert(dA_prev.shape == A_prev.shape)
    return dA_prev


def relu(Z):
    return np.maximum(0, Z)

def relu_backward(X):
    return np.array(X>0,dtype=np.float32)

def sofmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)


folder_path='./Data'
train_label=[]
num_unique_classes=[]
test_label=[]
idg_train = ImageDataGenerator(rescale=1./255,horizontal_flip=True,shear_range=0.2,zoom_range=0.2)
idg_test = ImageDataGenerator(rescale=1./255)

for file_name in os.listdir(folder_path):
    if file_name=='Training_set.csv' :
        train_label=pd.read_csv("./Data/"+file_name)
        num_unique_classes=len(train_label['label'].unique())
    if file_name == 'Testing_set.csv':
        test_label=pd.read_csv("./Data/"+file_name)


train_data= idg_train.flow_from_dataframe(dataframe=train_label,directory="./Data/train/",x_col='filename',y_col='label',target_size=(64,64),batch_size=32,class_mode='categorical')
test_data= idg_test.flow_from_dataframe(dataframe=test_label,directory="./Data/test/",x_col='filename',y_col=None,target_size=(64,64),batch_size=32,class_mode=None)

train_x,train_y=next(train_data)

def calculate_accuracy(y_true, y_pred):
    y_pred = y_pred.T
    y_pred_classes = np.argmax(y_pred, axis=0)
    y_true_classes = np.argmax(y_true, axis=0)

    accuracy = np.mean(y_pred_classes == y_true_classes)
    return accuracy

def model(train_x,train_y,epochs=100,learning_rate=0.001):
    W1=np.random.randn(3,3,3,16)*0.01
    B1=np.zeros((1,1,1,16))
    
    W2=np.random.randn(3,3,16,64)*0.01
    B2=np.zeros((1,1,1,64))
    
    W3=np.random.randn(128,16384)*0.01
    B3=np.zeros((128,1))
    
    W4=np.random.randn(15,128)*0.01
    B4=np.zeros((15,1))
    train_y=train_y.T
    
    for i in range(epochs):
        #Forward
        Z1,cache1=conv_forward(train_x,W1,B1,{'stride':1,'pad':1})
        Z2,cache2=pool_forward(Z1,{'f':2,'stride':2})
        
        Z3,cache3=conv_forward(Z2,W2,B2,{'stride':1,'pad':1})
        Z4,cache4=pool_forward(Z3,{'f':2,'stride':2})
        
        Z5=Z4.reshape(Z4.shape[0],-1).T
        m=Z5.shape[1]
        
        Z6=np.dot(W3,Z5)+B3
        A6=relu(Z6)

        Z7=np.dot(W4,A6)+B4
        A7=sofmax(Z7)
        accuracy=calculate_accuracy(train_y.T,A7)
        print('Epochs:',i,'Accuracy:',accuracy)
        #Backward
        dZ7=A7-train_y
        dw7=(-1/m)*np.dot(dZ7,A6.T)
        db7=(-1/m)*np.sum(dZ7,axis=1,keepdims=True)
        
        dz6=np.dot(W4.T,dZ7)*relu_backward(Z6)
        dw6=(-1/m)*np.dot(dz6,Z5.T)
        db6=(-1/m)*np.sum(dz6,axis=1,keepdims=True)
        
        dz5=np.dot(dz6.T,W3)
        dz5=dz5.reshape(Z4.shape).T
        
        dz4=pool_backward(dz5.T,cache4,mode='max')
        dz3,dw3,db3=conv_backward(dz4,cache3)
        dz2=pool_backward(dz3,cache2,mode='max')
        dz1,dw1,db1=conv_backward(dz2,cache1)
        #update
        W1=W1-learning_rate*dw1
        B1=B1-learning_rate*db1
        W2=W2-learning_rate*dw3
        B2=B2-learning_rate*db3
        W3=W3-learning_rate*dw6
        B3=B3-learning_rate*db6
        W4=W4-learning_rate*dw7
        B4=B4-learning_rate*db7
    return W1,B1,W2,B2,W3,B3,W4,B4
    
W1,B1,W2,B2,W3,B3,W4,B4=model(train_x,train_y,epochs=100,learning_rate=0.01)