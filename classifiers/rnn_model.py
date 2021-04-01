from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from numpy import asarray
from numpy import zeros
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import datetime

from configuration import config
from utils.metrics import *


class RNNClassifier:

    def __init__(self) -> None:
        pass

    def train(self, samples, targets):
        """
        Train and evaluate the BILSTM classifier.
        Parameters
        ----------
        samples
        targets
        """
        X = np.asarray(samples)
        Y = np.asarray(targets)

        print(X.shape)
        print(Y.shape)

        # Split dataset in train/test of 80/20 % accordingly
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(x_train)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_test = tokenizer.texts_to_sequences(x_test)

        vocab_size = len(tokenizer.word_index) + 1

        # Pad sequences to max length
        x_train = pad_sequences(x_train, padding='post', maxlen=config.props['embeddings_max_length'])
        x_test = pad_sequences(x_test, padding='post', maxlen=config.props['embeddings_max_length'])

        embeddings_dictionary = dict()

        # Load the word embeddings
        glove_file = open(config.props['embeddings'], encoding="utf8")

        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions
        glove_file.close()

        # Encode inputs using GloVe embeddings
        embedding_matrix = zeros((vocab_size, config.props['embeddings_shape']))
        for word, index in tokenizer.word_index.items():
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        print('Train set:', x_train.shape, y_train.shape)
        print('Test set:', x_test.shape, y_test.shape)

        # Build the RNN network
        model = Sequential()
        model.add(Input(shape=(config.props['embeddings_max_length'],)))
        model.add(Embedding(vocab_size, config.props['embeddings_shape'], weights=[embedding_matrix], trainable=False))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(y_test.shape[1], activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam')

        print(model.summary())

        # Early stopping mechanism to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(x_train, y_train, batch_size=128, epochs=60, verbose=1,
                            validation_split=0.2, callbacks=[early_stopping])

        score = model.evaluate(x_test, y_test, verbose=1)
        print('Final Score: ', score)

        predictions = model.predict(x_test)
        y_pred = self.probas_to_classes(predictions)

        # Zero predictor for comparison
        # self.calculate_performance(y_test, np.zeros_like(y_pred))
        # print('#'*36)

        self.calculate_performance(y_test, y_pred)
        self.plot_performance(history)

    @staticmethod
    def probas_to_classes(probabilities):
        """
        Returns OneHot array indicating if each label should be applied or not (if prob > 0.5)
        Parameters
        ----------
        probabilities: probabilities as predicted from the classifier

        Returns
        -------
        OneHot indicator array
        """
        return (probabilities > 0.5).astype('int32')

    @staticmethod
    def calculate_performance(y_true, y_pred):
        """
        Calculate BILSTM classifier performance using R, P, F1, @K metrics
        Parameters
        ----------
        y_true: targets
        y_pred: predictions
        """
        print(classification_report(y_true=y_true, y_pred=y_pred))

        print('@' * 36)

        for i in range(1, 11):
            r_k = mean_recall_k(y_true, y_pred, k=i)
            p_k = mean_precision_k(y_true, y_pred, k=i)
            rp_k = mean_rprecision_k(y_true, y_pred, k=i)
            ndcg_k = mean_ndcg_score(y_true, y_pred, k=i)

            print('R@{}: {:1.4f} P@{}: {:1.4f} RP@{}: {:1.4f} NDCG@{}: {:1.4f}'
                  .format(i, r_k, i, p_k, i, rp_k, i, ndcg_k))

    @staticmethod
    def plot_performance(history):
        """
        Plots training performance using validation loss
        Parameters
        ----------
        history: history object from keras model
        """
        best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1

        plt.figure(figsize=(10, 8))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.axvline(best_epoch, color='k', linestyle='dashed', linewidth=1,
                    label='Best epoch: {}'.format(best_epoch))
        plt.title('Model Loss Curve')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test', 'Best epoch: {}'.format(best_epoch)], loc='upper right')
        plt.savefig('resources/loss_curve_{}.png'.format(datetime.datetime.now()))
        plt.show()
