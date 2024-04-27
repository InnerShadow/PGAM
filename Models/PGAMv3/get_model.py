from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Reshape, Conv2D, MaxPooling2D, Flatten, GRU, Dropout, Attention
from keras.losses import SparseCategoricalCrossentropy

def get_model(n_window, vocab_size, embedding_size):
    input = Input(shape = (n_window, ))
    body = BatchNormalization()(input)
    body = Reshape((25, 10, 1))(body)
    body = Conv2D(64, (2, 2), padding = 'same', activation = 'elu')(body)
    body = MaxPooling2D((2, 2))(body)
    body = Dropout(0.7)(body)
    body = Conv2D(64, (2, 2), padding = 'same', activation = 'elu')(body)
    body = MaxPooling2D((2, 2))(body)
    body = Dropout(0.5)(body)
    body = Flatten()(body)
    body = Dense(512, activation = 'elu')(body)
    output = Dense(2, activation = 'softmax')(body)

    model = Model(inputs = input, outputs = output)

    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    model.summary()

    return model

