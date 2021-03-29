import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from nltk.tokenize import word_tokenize


class SVMClassifier:

    def __init__(self):
        pass

    def probas_to_classes(self, probabilities):
        return (probabilities > 0.5).astype('int32')

    def calculate_performance(self, network, x, y_true):

        predictions = network.predict_proba(x)
        y_pred = self.probas_to_classes(predictions)

        # Overall
        start, end = (0, 19)
        print('Overall evaluation')
        print('----------------------------------------------------')
        for average_type in ['micro', 'macro', 'weighted']:
            p = precision_score(y_true[:, start:end], y_pred[:, start:end], average=average_type)
            r = recall_score(y_true[:, start:end], y_pred[:, start:end], average=average_type)
            f1 = f1_score(y_true[:, start:end], y_pred[:, start:end], average=average_type)
            print('{:8} - Precision: {:1.4f}   Recall: {:1.4f}   F1: {:1.4f}'.format(average_type, p, r, f1))
        print('----------------------------------------------------')

    def train(self, samples, targets):
        text_clf = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 5), max_features=400000, tokenizer=word_tokenize)),
            ('tfidf', TfidfTransformer(use_idf=True)),
            ('clf', OneVsRestClassifier(LogisticRegression(multi_class='ovr', solver='sag', n_jobs=1,
                                        max_iter=100, verbose=0), n_jobs=8))
        ])

        print(text_clf.get_params().keys())

        # Run Classifier
        X = np.asarray(samples)
        Y = np.asarray(targets)

        print(X.shape)
        print(Y.shape)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

        print('Train set:', x_train.shape, y_train.shape)
        print('Validation set:', x_val.shape, y_val.shape)
        print('Test set:', x_test.shape, y_test.shape)

        text_clf.fit(x_train, y_train)

        # Evaluation Report
        print('VALIDATION EVALUATION')
        print('=====================')
        print(x_val.shape)
        print(y_val.shape)
        self.calculate_performance(text_clf, x_val, y_val)

        print('TEST EVALUATION')
        print('=====================')
        print(x_test.shape)
        print(y_test.shape)
        self.calculate_performance(text_clf, x_test, y_test)
