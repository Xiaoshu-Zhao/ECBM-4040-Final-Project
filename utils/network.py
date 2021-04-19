import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, MaxPooling2D, Concatenate,UpSampling2D
from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Embedding, Add, Bidirectional, Concatenate, RepeatVector, GRU


start = Input(shape=(224,224,1))
encoder = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2,2))(start)
encoder = Conv2D(128, (3, 3), activation='relu', padding='same', strides=(1,1))(encoder)
encoder = Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2,2))(encoder)
encoder = Conv2D(256, (3, 3), activation='relu', padding='same', strides=(1,1))(encoder)
encoder = Conv2D(256, (3, 3), activation='relu', padding='same', strides=(2,2))(encoder)
encoder = Conv2D(512, (3, 3), activation='relu', padding='same', strides=(1,1))(encoder)
encoder = Conv2D(512, (3, 3), activation='relu', padding='same', strides=(1,1))(encoder)
encoder = Conv2D(256, (3, 3), activation='relu', padding='same', strides=(1,1))(encoder)

feature_extractor = Input(shape=(28,28,1536))
fusion = Concatenate()([encoder,feature_extractor])
fusion = Conv2D(256, (1, 1), activation='relu', padding='same', strides=1)(fusion)

decoder = Conv2D(128, (3, 3), activation="relu", padding="same")(fusion)
decoder = UpSampling2D((2, 2))(decoder)
decoder = Conv2D(64, (3, 3), activation="relu", padding="same")(decoder)
decoder = Conv2D(64, (3, 3), activation="relu", padding="same")(decoder)
decoder = UpSampling2D((2, 2))(decoder)
decoder = Conv2D(32, (3, 3), activation="relu", padding="same")(decoder)
decoder = Conv2D(2, (3, 3), activation="tanh", padding="same")(decoder)
decoder = UpSampling2D((2, 2))(decoder)

deep_color = Model([start,feature_extractor],decoder)

deep_color.compile(optimizer='Adam',loss='mse',metrics=['accuracy'])

deep_color.summary()