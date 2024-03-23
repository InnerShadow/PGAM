from keras.models import Model
from keras.layers import LSTM, Dense, Embedding, Input, BatchNormalization, Dropout


def get_model(n_window, vocab_size, embedding_size):
    input = Input(shape = (n_window, ))
    body = Embedding(input_dim = vocab_size, output_dim = embedding_size)(input)
    body = LSTM(4, return_sequences = True)(body)
    body = BatchNormalization()(body)
    body = Dropout(0.5)(body)
    # body = LSTM(64, return_sequences = True)(body)
    # body = BatchNormalization()(body)
    # body = Dropout(0.5)(body)
    # body = LSTM(64, return_sequences = True)(body)
    # body = BatchNormalization()(body)
    # body = Dropout(0.5)(body)
    # body = LSTM(64, return_sequences = True)(body)
    # body = BatchNormalization()(body)
    # body = Dropout(0.5)(body)
    body = LSTM(4, return_sequences = False)(body)
    body = BatchNormalization()(body)
    body = Dropout(0.5)(body)
    output = Dense(2, activation = 'softmax')(body)

    model = Model(inputs = input, outputs = output)
    
    model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

    model.summary()

    return model

