# Importing necessary libraries and modules
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

class SentimentModel:
    # Maximum number of unique words in the vocabulary
    MAX_FEATURES = 10000
    # Maximum length of input sequences after padding
    MAX_LEN = 200
    # Dimensionality of the word embedding
    EMBEDDING_DIM = 100
    # Number of filters in the convolutional layer
    FILTERS = 128
    # Size of the convolutional kernel
    KERNEL_SIZE = 5
    # Number of units in the hidden layer
    HIDDEN_DIMS = 64
    # Batch size for training
    BATCH_SIZE = 32
    # Number of training epochs
    EPOCHS = 5

    def __init__(self):
        # Initializes the SentimentModel object and builds the underlying CNN model
        self.model = self._build_model()

    def _build_model(self):
        # Builds the CNN model architecture
        model = Sequential([
            Embedding(self.MAX_FEATURES, self.EMBEDDING_DIM, input_length=self.MAX_LEN),
            Conv1D(self.FILTERS, self.KERNEL_SIZE, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(self.HIDDEN_DIMS, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, x_test, y_test):
        # Trains the model on the provided training data
        x_train = pad_sequences(x_train, maxlen=self.MAX_LEN)
        x_test = pad_sequences(x_test, maxlen=self.MAX_LEN)
        self.model.fit(x_train, y_train, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, validation_data=(x_test, y_test))

    def evaluate(self, x_test, y_test):
        # Evaluates the model on the provided test data
        x_test = pad_sequences(x_test, maxlen=self.MAX_LEN) 
        loss, accuracy = self.model.evaluate(x_test, y_test)
        print(f"Test loss: {loss:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")

    def predict(self, texts):
        # Predicts sentiment labels for the given list of texts
        word_index = imdb.get_word_index()
        sequences = [self._text_to_sequence(text, word_index) for text in texts]
        sequences = pad_sequences(sequences, maxlen=self.MAX_LEN)
        predictions = self.model.predict(sequences)
        return ["Positive" if pred > 0.5 else "Negative" for pred in predictions]

    @staticmethod
    def _text_to_sequence(text, word_index):
        # Converts text to a sequence of word indices
        return [word_index.get(word, 0) + 3 for word in text.split()]

if __name__ == "__main__":
    # Loading the IMDb movie reviews dataset
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=SentimentModel.MAX_FEATURES)
    
    # Creating an instance of SentimentModel
    sentiment_model = SentimentModel()
    # Training the model
    sentiment_model.train(x_train, y_train, x_test, y_test)
    # Evaluating the model
    sentiment_model.evaluate(x_test, y_test)
    # Making predictions on new texts
    new_texts = ["I really enjoyed this movie!" , "This film is terrible.", "The acting was superb."]
    predictions = sentiment_model.predict(new_texts)
    
    # Displaying predictions
    for text, sentiment in zip(new_texts, predictions):
        print(f"Text: {text}\nSentiment: {sentiment}")
