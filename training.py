# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

# Train keras model to detect sentiment in tweets.
import csv
import numpy
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download training data from (https://www.kaggle.com/kazanova/sentiment140).
# Download word embeddings from (https://www.kaggle.com/danielwillgeorge/glove6b100dtxt).

# Parameters.
embedding_dim = 100
text_length = 16
epochs = 10
batchSize = 64
training_size=1000000

# Sentences and labels.
sentences=[]
labels=[]

# Get data from csv file.
with open("./training.1600000.processed.noemoticon.csv", encoding="utf8") as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        sentences.append(row[5])
        label=row[0]
        if label=='0':
          labels.append(0)
        else:
          labels.append(1)   


# Create a tokenizer.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
token_size=len(word_index)


# Create sequences.
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=text_length, padding="post", truncating="post")

# Training and validation data.
training_data = numpy.array(padded[0:training_size])
training_labels = numpy.array(labels[0:training_size])
validation_data = numpy.array(padded[training_size:])
validation_labels = numpy.array(labels[training_size:])  

# Get 100 dimension version of GloVe from Stanford.
embeddings_index = {}
with open('./glove.6B.100d.txt', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = numpy.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embeddings_matrix = numpy.zeros((token_size+1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector

# Create model with 1 output unit for classification.
model = keras.Sequential([
    keras.layers.Embedding(token_size+1, embedding_dim, input_length=text_length, weights=[embeddings_matrix], trainable=False),
    keras.layers.Dropout(0.2),
    keras.layers.Conv1D(64, 5, activation="relu"),
    keras.layers.MaxPooling1D(pool_size=4),
    keras.layers.LSTM(64),
    keras.layers.Dense(1, activation="sigmoid")
])

# Set loss function and optimizer.
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      # Stop when validation accuracy is more than 98%.
        if logs.get('val_accuracy') is not None and logs.get('val_accuracy') > 0.98:
            print("\nTraining Stopped!")
            self.model.stop_training = True


# Callback function to check accuracy.
checkAccuracy = myCallback()

# Train model.
model.fit(training_data, training_labels, epochs=epochs, callbacks=[
          checkAccuracy], batch_size=batchSize, validation_data=(validation_data, validation_labels), verbose=1)