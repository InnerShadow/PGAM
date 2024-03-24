from keras.models import Model
from keras.layers import LSTM, Dense, Embedding, Input, BatchNormalization, Dropout, concatenate

def get_model(n_window, vocab_size, embedding_size):
    input_1 = Input(shape = (n_window, ))
    body = Embedding(input_dim = vocab_size, output_dim = embedding_size)(input_1)
    body = LSTM(64, return_sequences = True)(body)
    body = BatchNormalization()(body)
    body = Dropout(0.5)(body)
    body = LSTM(64, return_sequences = True)(body)
    body = BatchNormalization()(body)
    body = Dropout(0.5)(body)
    body = LSTM(64, return_sequences = True)(body)
    body = BatchNormalization()(body)
    body = Dropout(0.5)(body)
    body = LSTM(64, return_sequences = True)(body)
    body = BatchNormalization()(body)
    body = Dropout(0.5)(body)
    output_1 = LSTM(64, return_sequences = False)(body)

    input_2 = Input(shape = (n_window, ))
    body = Embedding(input_dim = vocab_size, output_dim = embedding_size)(input_2)
    body = LSTM(64, return_sequences = True)(body)
    body = BatchNormalization()(body)
    body = Dropout(0.5)(body)
    body = LSTM(64, return_sequences = True)(body)
    body = BatchNormalization()(body)
    body = Dropout(0.5)(body)
    body = LSTM(64, return_sequences = True)(body)
    body = BatchNormalization()(body)
    body = Dropout(0.5)(body)
    body = LSTM(64, return_sequences = True)(body)
    body = BatchNormalization()(body)
    body = Dropout(0.5)(body)
    output_2 = LSTM(64, return_sequences = False)(body)

    merged = concatenate([output_1, output_2])

    output = Dense(2, activation = 'softmax')(merged)

    model = Model(inputs = [input_1, input_2], outputs = output)
    
    model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

    model.summary()

    return model

