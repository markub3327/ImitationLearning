import numpy as np
import math

from tensorflow.keras.layers import Input, Dense, Dropout, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from nets.base_model import BaseModel

def decision_block(units, x, id):
    l = Dense(units, activation='swish', use_bias=True, kernel_initializer='he_uniform', name='FC_'+id)(x)     # new activation function
    l = Dropout(0.5, name='FC_'+id+'_dropout')(l)
    return l

class Agent:
    def __init__(self, lr=0.0001):
        self.lr = lr

    def create(self, input_shape, hid=[32, 64]):
        state_inp = Input(shape=input_shape, name='image_input')

        # Base model (for transfer learning)
        l1, self._base_model = BaseModel(state_inp)

        # Memory blocks
        l2 = GRU(hid[0], use_bias=True, name='gru_1')(l1)

        # Decision blocks
        l3 = l2
        for i, x in enumerate(hid[1:]):
            l3 = decision_block(x, l3, str(i))

        # output layer
        steering_layer = Dense(1, activation='tanh', use_bias=True, name="steering")(l3)
        acceleration_brake_layer = Dense(2, activation='sigmoid', use_bias=True, name="acceleration_brake")(l3)

        # create model
        self._model = Model(inputs=state_inp, outputs=[steering_layer, acceleration_brake_layer], name="agent")

    def load(self, path):
        self._model = load_model(path)
        # show model
        self._model.summary()

    def train(self, states, actions, epochs=100, batch_size=64, top_only=True, callbacks=[]):
        if (top_only == False):
            self._base_model.trainable = True

            # Let's take a look to see how many layers are in the base model
            print("Number of layers in the base model: ", len(self._base_model.layers))

            # Fine-tune from this layer onwards
            fine_tune_at = 100

            # Freeze all the layers before the `fine_tune_at` layer
            for layer in self._base_model.layers[:fine_tune_at]:
                layer.trainable =  False
            self._model.compile(optimizer=Adam(learning_rate=self.lr/10.0, amsgrad=True), loss='mse')

            # show model
            self._model.summary()
        else:
            # Freeze the base model
            self._base_model.trainable = False

            self._model.compile(optimizer=Adam(learning_rate=self.lr, amsgrad=True), loss='mse')

            # show model
            self._model.summary()

        self._model.fit(x=states, y={"steering": actions[:, 0], "acceleration_brake": actions[:, 1:]}, batch_size=batch_size, epochs=epochs, verbose=1, \
             validation_split=0.20, callbacks=[EarlyStopping(monitor='val_loss', patience=16, verbose=1, restore_best_weights=True)]+callbacks)
    
    def save(self, path='save/model.h5'):
        self._model.save(path)
        print(f"Model saved to {path}")
    
    def save_plot(self, path='model.png'):
        plot_model(self._model, to_file=path, show_shapes=True)

    def act(self, obs):
        obs = np.expand_dims(obs, axis=0)
        return self._model(obs)
