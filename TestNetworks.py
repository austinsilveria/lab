import keras
from keras.utils import plot_model
import Networks as nt
import numpy as np


def test_SC2FullConv():
    non_spatial1 = np.random.rand(32, 64, 64, 3)
    screen1 = np.random.rand(32, 64, 64, 17)
    minimap1 = np.random.rand(32, 64, 64, 7)
    print('Sample non-spatial data:', non_spatial1.shape)
    print('Sample screen data:', screen1.shape)
    print('Sample minimap data:', minimap1.shape)
    model = nt.Networks().SC2FullConv()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy')
    output = model.predict([non_spatial1, screen1, minimap1])
    model.summary()
    plot_model(model, show_shapes=True, to_file='model.png')
    print('Screen1 output:', output[0].shape)
    print('Screen2 output:', output[1].shape)
    print('Minimap output:', output[2].shape)


if __name__ == '__main__':
    test_SC2FullConv()
