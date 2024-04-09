from keras.models import Model
from keras.layers import LSTM, Dense, Embedding, Input, Conv1D, MaxPooling1D, Flatten, BatchNormalization

def get_model(n_window, vocab_size, embedding_size):
    input = Input(shape = (n_window, ))
    # body = Embedding(input_dim = vocab_size, output_dim = embedding_size)(input)
    body = BatchNormalization()(input)
    body = Dense(1024 * 2, activation = 'elu')(body)
    body = Dense(512 * 2, activation = 'elu')(body)
    body = Dense(256 * 2, activation = 'elu')(body)
    body = Dense(128, activation = 'elu')(body)
    body = Dense(128, activation = 'elu')(body)
    body = Dense(128, activation = 'elu')(body)
    body = Dense(128, activation = 'elu')(body)
    body = Dense(128, activation = 'elu')(body)
    body = Dense(128, activation = 'elu')(body)
    body = Dense(128, activation = 'elu')(body)
    output = Dense(2, activation = 'softmax')(body)

    model = Model(inputs = input, outputs = output)

    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    model.summary()

    return model

