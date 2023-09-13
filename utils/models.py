from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

kernel_init_1 = 'glorot_uniform'
kernel_init_2 = 'he_uniform'


class ModelVad1:
    def __new__(cls, window_size: int):
        model = Sequential()

        model.add(LSTM(
                window_size * 3,
                name='in'
                # return_sequences=True,
                # dropout=.2
            ))
        model.add(Dense(window_size * 2, name='d1', activation='relu', kernel_initializer=kernel_init_1))
        model.add(Dense(int(window_size * 1), name='d2', activation='relu', kernel_initializer=kernel_init_1))
        model.add(Dense(int(window_size * .5), name='d3', activation='relu', kernel_initializer=kernel_init_1))
        model.add(Dense(int(window_size * .2), name='d4', activation='relu', kernel_initializer=kernel_init_1))
        model.add(Dense(1, name='out'))
        return model


class ModelVad2:
    def __new__(cls, window_size: int):
        model = Sequential()

        model.add(LSTM(units=128, stateful=False, return_sequences=True))
        model.add(LSTM(units=64, stateful=False, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=32))
        model.add(Dropout(0.2))
        model.add(Dense(20))
        return model


class ModelMax1:
    def __new__(cls, window_size: int):
        model = Sequential()
        model.add(LSTM(
            45,
        ))
        model.add(Dense(90, activation='sigmoid'))
        model.add(Dense(180, activation='sigmoid'))
        model.add(Dense(360, activation='sigmoid'))
        model.add(Dense(150, activation='sigmoid'))
        model.add(Dense(1, activation='linear'))
        return model


class Model2LSTM:
    def __new__(cls, window_size: int):
        model = Sequential()
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(64, input_shape=(128, 1)))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(4, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1))
        return model
