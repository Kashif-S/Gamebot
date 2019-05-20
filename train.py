import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, CuDNNLSTM, Dropout, Flatten, CuDNNGRU, LSTM, ConvLSTM2D, TimeDistributed, Conv2D, MaxPooling2D
from tensorflow.keras.utils import normalize
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import CuDNNLSTM

rate_dropout = 0.2
memory_size = 10

training_data = np.load('training_data.npz')['training_data']

X, y = zip(*training_data)

X = np.array([X[(i-memory_size):i] for i in range(memory_size, len(X))], np.int8)
y = np.array(y[memory_size:], np.int8)
print(np.shape(X), np.shape(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

X_train = X_train / 255.0
X_test = X_test / 255.0

def create_model():
    model = Sequential()

    # model.add(CuDNNLSTM(2048, input_shape=(160, 240), return_sequences=True))
    # model.add(Dropout(rate_dropout))
    # model.add(CuDNNLSTM(2048))
    # model.add(Dropout(rate_dropout))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(12, activation='softmax'))

    model.add(ConvLSTM2D(16, (5, 5), input_shape=(memory_size, 80, 120, 1), activation='relu', data_format='channels_last', return_sequences=False))
    model.add(Dropout(rate_dropout))
    model.add(Flatten())
    #model.add(Dense(1024, activation='relu'))
    #model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(12, activation='softmax'))

    #model.add(TimeDistributed(Conv2D(128, (5, 5), activation='relu', data_format='channels_first'), input_shape=(5, 1, 40, 60)))
    #model.add(TimeDistributed(MaxPooling2D(pool_size=(5, 5))))
    #model.add(TimeDistributed(Flatten()))
    #model.add(CuDNNLSTM(512))
    #model.add(Dropout(rate_dropout))
    #model.add(Dense(1024, activation='relu'))
    #model.add(Dense(512, activation='relu'))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dense(128, activation='relu'))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dense(12, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    return model

def run():
    model = create_model()
    model.fit(X, y, batch_size=100, epochs=3, validation_data=(X_test, y_test))
    model.save('model.h5')

run()