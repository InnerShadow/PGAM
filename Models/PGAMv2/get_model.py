from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Reshape, Conv2D, MaxPooling2D, Flatten, concatenate

def get_model(n_window, vocab_size, embedding_size):
    input = Input(shape = (n_window, ))
    body = Reshape(target_shape = (40, 25, 1))(input)
    body = BatchNormalization()(body)
    body = Conv2D(64, (3, 3), padding = 'same', activation = 'elu')(body)
    body_1 = MaxPooling2D((2 , 2))(body)
    body = Conv2D(64, (3, 3), padding = 'same', activation = 'elu')(body_1)
    body_2 = MaxPooling2D((2 , 2))(body)
    body = Conv2D(64, (3, 3), padding = 'same', activation = 'elu')(body_2)
    body_3 = MaxPooling2D((2 , 2))(body)
    body_1 = Flatten()(body_1)
    body_2 = Flatten()(body_2)
    body_3 = Flatten()(body_3)
    concat = concatenate([body_1, body_2, body_3])
    body = Dense(1024, activation = 'elu')(concat)
    body = Dense(512, activation = 'elu')(body)
    body = Dense(128, activation = 'elu')(body)
    output = Dense(2, activation = 'softmax')(body)

    model = Model(inputs = input, outputs = output)

    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    model.summary()

    return model

