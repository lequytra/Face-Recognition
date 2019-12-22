from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

def SVM():
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1), W_regularizer=l2(0.01))
    model.add(Activation('softmax'))
    model.compile(loss='squared_hinge',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model