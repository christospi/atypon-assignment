from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from utils.metrics import *


class LogRegClassifier:

    def __init__(self) -> None:
        pass

    def train(self, samples, targets):
        """
        Train and evaluate the Logistic Regression classifier.
        Parameters
        ----------
        samples
        targets
        """

        # Available solvers (sag): ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
        text_clf = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 5), max_features=50000, tokenizer=word_tokenize)),
            ('tfidf', TfidfTransformer(use_idf=True)),
            ('clf', OneVsRestClassifier(LogisticRegression(multi_class='ovr', solver='sag', n_jobs=1,
                                        max_iter=100, verbose=0), n_jobs=8))
        ])

        X = np.asarray(samples)
        Y = np.asarray(targets)

        print(X.shape)
        print(Y.shape)

        # Split dataset in train/val/test of 80/10/10 % accordingly
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

        print('Train set:', x_train.shape, y_train.shape)
        print('Validation set:', x_val.shape, y_val.shape)
        print('Test set:', x_test.shape, y_test.shape)
        print(text_clf.get_params().keys())

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

    def calculate_performance(self, network, x, y_true):
        """
        Calculate Logistic Regression performance using R, P, F1, @K metrics
        Parameters
        ----------
        network: classifier
        x: samples
        y_true: targets
        """
        predictions = network.predict_proba(x)
        y_pred = self.probas_to_classes(predictions)

        # Overall
        print('Overall evaluation')
        print('----------------------------------------------------')
        for average_type in ['micro', 'macro', 'weighted']:
            p = precision_score(y_true, y_pred, average=average_type)
            r = recall_score(y_true, y_pred, average=average_type)
            f1 = f1_score(y_true, y_pred, average=average_type)
            print('{:8} - Precision: {:1.4f}   Recall: {:1.4f}   F1: {:1.4f}'.format(average_type, p, r, f1))
        print('----------------------------------------------------')

        print(classification_report(y_true=y_true, y_pred=y_pred))

        print('@'*36)

        for i in range(1, 11):
            r_k = mean_recall_k(y_true, y_pred, k=i)
            p_k = mean_precision_k(y_true, y_pred, k=i)
            rp_k = mean_rprecision_k(y_true, y_pred, k=i)
            ndcg_k = mean_ndcg_score(y_true, y_pred, k=i)

            print('R@{}: {:1.4f} P@{}: {:1.4f} RP@{}: {:1.4f} NDCG@{}: {:1.4f}'
                  .format(i, r_k, i, p_k, i, rp_k, i, ndcg_k))