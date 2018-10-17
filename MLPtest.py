from tensorflow.contrib.keras.api.keras import backend
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import Input
from tensorflow.python.keras.engine import Model
# from tensorflow.python.keras.
from tensorflow.python.keras.layers import Dropout, BatchNormalization
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from data import getData
from tensorflow.python.keras.layers.merge import add, Add


def block(x):
    xprime = BatchNormalization()(x)

    xprime = Dense(units=128, activation="relu")(xprime)
    xprime = Dropout(0.5)(xprime)

    xprime = Dense(units=128, activation="relu")(xprime)
    xprime = Dropout(0.5)(xprime)

    return xprime


inputs = Input([25])
xp = Dense(units=128, activation="relu")(inputs)
x = Dropout(0.5)(xp)
x = block(x)
x = block(x)
x = block(x)
x = block(x)
x = block(x)
regression = Dense(units=1)(x)

model = Model(inputs=inputs, outputs=regression)

model.summary()
with open("model.yaml", 'w') as m:
    m.write(model.to_yaml())
with open("model.json", 'w') as m:
    m.write(model.to_json())
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


model.compile(optimizer="rmsprop", loss="mean_squared_error", metrics=[rmse])
trainingX, trainingY, testingX, testingY = getData()

hist = model.fit(x=trainingX, y=trainingY, batch_size=1024, epochs=10, validation_split=0.2)

score = model.evaluate(testingX, testingY, batch_size=1024)
#
print("testing result:", score)
