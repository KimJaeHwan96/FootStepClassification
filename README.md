<h2>2020년 1학기 인공지능 수업 프로젝트입니다</h2>

3명이 한 조를 이뤘으며 데이터 수집, 데이터 전처리, 신경망 학습으로 나누어 임무 분담을 하였고 저는 신경망 학습을 담당하였습니다.   

데이터는 사람이 걷는 동영상으로 수집을 하였고       

<img src="https://user-images.githubusercontent.com/64777061/94391151-bf844580-018f-11eb-8caf-c3e7a0435762.png" width="30%" height="40%"></img>   


이렇게 동영상을 흑백 이미지로 데이터를 전처리 하였습니다.   
datasets에는 28명의 서로 다른 걸음걸이 데이터가 10장씩 총 280장 있습니다.   
이 데이터들을 252장의 train 데이터와 28장의 test 데이터로 나눴습니다.   

먼저 순차적으로 신경망을 쌓기위해 Sequential 클래스의 객체를 만듭니다.   
  model = Sequential()   

- - -

  model.add(Conv2D(128, kernel_size = (3, 3), padding='valid', activation='relu', input_shape=(150, 272, 1)))   

3x3x128 커널로 convolution 하였고 활성함수로 sigmoid나 tanh대신 relu를 사용하여 back propagation 할때   


  model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))


  model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
  model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(28, activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
