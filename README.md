<h2>2020년 1학기 인공지능 수업 프로젝트입니다</h2>

3명이 한 조를 이뤘으며 데이터 수집, 데이터 전처리, 신경망 학습으로 나누어 임무 분담을 하였고 저는 신경망 학습을 담당하였습니다.   

데이터는 사람이 걷는 동영상으로 수집을 하였고       

<img src="https://user-images.githubusercontent.com/64777061/94391151-bf844580-018f-11eb-8caf-c3e7a0435762.png" width="30%" height="40%"></img>   


이렇게 동영상을 흑백 이미지로 데이터를 전처리 하였습니다.   
datasets에는 28명의 서로 다른 걸음걸이 데이터가 10장씩 총 280장 있습니다.   
이 데이터들을 252장의 train 데이터와 28장의 test 데이터로 나눴습니다.   


이러한 데이터들을 학습시키기 위해 신경망을 만들어야하는데 
순차적으로 신경망을 쌓기위해 Sequential 클래스로 객체를 만듭니다. 

    model = Sequential()   


128, kernel_size=(3,3) => kernel을 3x3x128으로 생성. padding='valid' => zero padding을 사용.
activation='relu' => 활성함수로 relu를 씀.  input_shape=(150, 272, 1) => 이미지는 150x272 크기, 흑백으로 받음.

    model.add(Conv2D(128, kernel_size = (3, 3), padding='valid', activation='relu', input_shape=(150, 272, 1)))   



max pooling을 사용하여 이미지 크기를 줄입니다.
 
    model.add(MaxPooling2D(pool_size=(2, 2)))



dropout을 사용하여 overfitting을 방지합니다
  
    model.add(Dropout(0.25))



Flatten()을 이용하여 2차원의 데이터를 1차원으로 만들어줍니다.

    model.add(Flatten())



 1차원의 데이터들을 28개로 만들고 활성함수인 softmax를 이용하여 각 클래스의 확률을 계산해줍니다.

    model.add(Dense(28, activation='softmax'))



손실함수로 cross_entropy를 사용하면 총합이 1인 각 클래스의 확률이 계산됩니다.     

각 파라미터마다 다른 크기의 업데이트를 적용하는 adam(Adaptive Moment Estimation)으로 최적화 해줍니다.    


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    




위의 코드들을 작성하면 다음과 같은 신경망을 쌓게 됩니다.    


<img src="https://user-images.githubusercontent.com/64777061/96664888-b05b7680-138e-11eb-8612-f61f0ffb6813.PNG" width="40%" height="20%"></img>



신경마

- - -






    Found 252 images belonging to 28 classes.
    Found 28 images belonging to 28 classes.
    Epoch 1/10
    84/84 [==============================] - 93s 1s/step - loss: 3.3165 - accuracy: 0.0675 - val_loss: 3.2166 - val_accuracy: 0.0357
    Epoch 2/10
    84/84 [==============================] - 92s 1s/step - loss: 1.9647 - accuracy: 0.4365 - val_loss: 0.0297 - val_accuracy: 0.9286
    Epoch 3/10
    84/84 [==============================] - 92s 1s/step - loss: 0.4996 - accuracy: 0.8571 - val_loss: 0.0018 - val_accuracy: 0.9286
    Epoch 4/10
    84/84 [==============================] - 92s 1s/step - loss: 0.2036 - accuracy: 0.9444 - val_loss: 0.0101 - val_accuracy: 0.9643
    Epoch 5/10
    84/84 [==============================] - 92s 1s/step - loss: 0.1024 - accuracy: 0.9563 - val_loss: 2.8877e-05 - val_accuracy: 1.0000
    Epoch 6/10
    84/84 [==============================] - 92s 1s/step - loss: 0.1243 - accuracy: 0.9524 - val_loss: 8.5012e-05 - val_accuracy: 0.9643
    Epoch 7/10
    84/84 [==============================] - 93s 1s/step - loss: 0.0912 - accuracy: 0.9563 - val_loss: 0.0013 - val_accuracy: 1.0000
    Epoch 8/10
    84/84 [==============================] - 92s 1s/step - loss: 0.1992 - accuracy: 0.9444 - val_loss: 0.0029 - val_accuracy: 0.9643
    Epoch 9/10
    84/84 [==============================] - 92s 1s/step - loss: 0.0976 - accuracy: 0.9762 - val_loss: 3.8743e-06 - val_accuracy: 1.0000
    Epoch 10/10
    84/84 [==============================] - 94s 1s/step - loss: 0.0936 - accuracy: 0.9722 - val_loss: 2.7417e-05 - val_accuracy: 1.0000
    -- Evaluate --
    accuracy: 100.00%
    -- Predict --
    {'1': 0, '10': 1, '11': 2, '12': 3, '13': 4, '14': 5, '15': 6, '16': 7, '17': 8, '18': 9, '19': 10, '2': 11, '20': 12, '21': 13, '22': 14, '23': 15, '24': 16, '25': 17,   
    '26': 18, '27': 19, '28': 20, '3': 21, '4': 22, '5': 23, '6': 24, '7': 25, '8': 26, '9': 27}
     ['1\\1_10.bmp', '10\\10_10.bmp', '11\\11_10.bmp', '12\\12_10.bmp', '13\\13_10.bmp', '14\\14_10.bmp', '15\\15_10.bmp', '16\\16_10.bmp', '17\\17_10.bmp', '18\\18_10.bmp',
    '19\\19_10.bmp', '2\\2_10.bmp', '20\\20_10.bmp', '21\\21_10.bmp', '22\\22_10.bmp', '23\\23_10.bmp', '24\\24_10.bmp', '25\\25_10.bmp', '26\\26_10.bmp', '27\\27_10.bmp',
    '28\\28_10.bmp', '3\\3_10.bmp', '4\\4_10.bmp', '5\\5_10.bmp', '6\\6_10.bmp', '7\\7_10.bmp', '8\\8_10.bmp', '9\\9_10.bmp']



위에 구축한 신경망을 train데이터로 학습하여 test데이터로 예측한 결과입니다.
