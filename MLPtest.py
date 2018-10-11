from tensorflow.contrib.keras.api.keras import backend
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import Input
from tensorflow.python.keras.engine import Model
# from tensorflow.python.keras.
from tensorflow.python.keras.layers import Dropout, BatchNormalization
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from data import getData

inputs = Input([25])
x = Dense(units=128, activation="relu")(inputs)
x = Dropout(0.5)(x)

x = Dense(units=128, activation="relu")(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

x = Dense(units=128, activation="relu")(x)
x = Dropout(0.5)(x)

x = Dense(units=128, activation="relu")(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

x = Dense(units=128, activation="relu")(x)
x = Dropout(0.5)(x)

x = Dense(units=128, activation="relu")(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

x = Dense(units=128, activation="relu")(x)
x = Dropout(0.5)(x)

x = Dense(units=128, activation="relu")(x)
x = Dropout(0.5)(x)

regression = Dense(units=1)(x)

model = Model(inputs=inputs, outputs=regression)


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


model.compile(optimizer="rmsprop", loss="mean_squared_error", metrics=[rmse])
trainingX, trainingY, testingX, testingY = getData()

hist = model.fit(x=trainingX, y=trainingY, batch_size=1024, epochs=10, validation_split=0.2)

score = model.evaluate(testingX, testingY, batch_size=1024)

print("testing result:", score)
