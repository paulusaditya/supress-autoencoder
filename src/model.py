from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D

def build_autoencoder():
    inp = Input(shape=(128, 128, 3))
    x = Conv2D(64, 3, activation='relu', padding='same')(inp)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    out = Conv2D(3, 3, activation='sigmoid', padding='same')(x)
    return Model(inp, out)
