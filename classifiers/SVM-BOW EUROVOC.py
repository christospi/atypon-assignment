import glob, json
import pickle
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

with open('/home/ichalkidis/nomothesia.nlp.training/nomothesia_nlp/data/datasets/eurovoc_en/eurovoc_en.index',
          'rb') as file:
    label_ids = pickle.load(file)


def parse_dataset(dataset):
    x = []
    filenames = glob.glob(dataset)
    y = np.zeros((len(filenames), 4271), dtype=np.int32)
    for i, filename in enumerate(filenames):
        with open(filename) as file:
            data = json.load(file)
            x.append(data['header'] + '\n' + data['recitals'] + '\n'.join(data['main_body']) + '\n' + data['attachments'])
            for j, tag in enumerate(data['concepts']):
                if tag in label_ids:
                    y[i][label_ids[tag]] = 1
    return np.asarray(x), y


def mean_precision_k(y_true, y_score, k=10):
    """Mean precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean precision @k : float
    """

    p_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            p_ks.append(ranking_precision_score(y_t, y_s, k=k))

    return np.mean(p_ks)


def mean_recall_k(y_true, y_score, k=10):
    """Mean recall at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean recall @k : float
    """

    r_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            r_ks.append(ranking_recall_score(y_t, y_s, k=k))

    return np.mean(r_ks)


def mean_ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    Mean NDCG @k : float
    """

    ndcg_s = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            ndcg_s.append(ndcg_score(y_t, y_s, k=k, gains=gains))

    return np.mean(ndcg_s)


def mean_rprecision_k(y_true, y_score, k=10):
    """Mean precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean precision @k : float
    """

    p_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            p_ks.append(ranking_rprecision_score(y_t, y_s, k=k))

    return np.mean(p_ks)


def ranking_recall_score(y_true, y_score, k=10):
    # https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf
    """Recall at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    return float(n_relevant) / n_pos


def ranking_precision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    return float(n_relevant) / k


def ranking_rprecision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(k, n_pos)


def average_precision_score(y_true, y_score, k=10):
    """Average precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    average precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1][:min(n_pos, k)]
    y_true = np.asarray(y_true)[order]

    score = 0
    for i in range(len(y_true)):
        if y_true[i] == pos_label:
            # Compute precision up to document i
            # i.e, percentage of relevant documents up to document i.
            prec = 0
            for j in range(0, i + 1):
                if y_true[j] == pos_label:
                    prec += 1.0
            prec /= (i + 1.0)
            score += prec

    if n_pos == 0:
        return 0

    return score / n_pos


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best


# Alternative API.

def dcg_from_ranking(y_true, ranking):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    ranking : array-like, shape = [k]
        Document indices, i.e.,
            ranking[0] is the index of top-ranked document,
            ranking[1] is the index of second-ranked document,
            ...
    k : int
        Rank.
    Returns
    -------
    DCG @k : float
    """
    y_true = np.asarray(y_true)
    ranking = np.asarray(ranking)
    rel = y_true[ranking]
    gains = 2 ** rel - 1
    discounts = np.log2(np.arange(len(ranking)) + 2)
    return np.sum(gains / discounts)


def ndcg_from_ranking(y_true, ranking):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    ranking : array-like, shape = [k]
        Document indices, i.e.,
            ranking[0] is the index of top-ranked document,
            ranking[1] is the index of second-ranked document,
            ...
    k : int
        Rank.
    Returns
    -------
    NDCG @k : float
    """
    k = len(ranking)
    best_ranking = np.argsort(y_true)[::-1]
    best = dcg_from_ranking(y_true, best_ranking[:k])
    return dcg_from_ranking(y_true, ranking) / best


def probas_to_classes(probabilities):
    return (probabilities > 0.5).astype('int32')


def calculate_performance(network, x, y_true):

    predictions = network.predict_proba(x)
    y_pred = probas_to_classes(predictions)

    template = 'R@{} : {:1.3f}   P@{} : {:1.3f}   RP@{} : {:1.3f}   NDCG@{} : {:1.3f}'

    # Overall
    for labels_range, frequency, message in zip([(0, 4271), (0, 746), (746, 4107), (4107, 4271)], ['Overall', 'Frequent', 'Few', 'Zero'],
                                                ['Overall', 'Frequent Labels (>=50 Occurrences in train set)',
                                                 'Few-shot (<=50 Occurrences in train set)', 'Zero-shot (No Occurrences in train set)']):
        start, end = labels_range
        print(message)
        print('----------------------------------------------------')
        for average_type in ['micro', 'macro', 'weighted']:
            p = precision_score(y_true[:, start:end], y_pred[:, start:end], average=average_type)
            r = recall_score(y_true[:, start:end], y_pred[:, start:end], average=average_type)
            f1 = f1_score(y_true[:, start:end], y_pred[:, start:end], average=average_type)
            print('{:8} - Precision: {:1.4f}   Recall: {:1.4f}   F1: {:1.4f}'.format(average_type, p, r, f1))

        for i in range(1, 11):
            r_k = mean_recall_k(y_true[:, start:end], predictions[:, start:end], k=i)
            p_k = mean_precision_k(y_true[:, start:end], predictions[:, start:end], k=i)
            rp_k = mean_rprecision_k(y_true[:, start:end], predictions[:, start:end], k=i)
            ndcg_k = mean_ndcg_score(y_true[:, start:end], predictions[:, start:end], k=i)
            print(template.format(i, r_k, i, p_k, i, rp_k, i, ndcg_k))
        print('----------------------------------------------------')


# SetUp Pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from nltk.tokenize import word_tokenize
text_clf = Pipeline([
     ('vect', CountVectorizer(ngram_range=(1, 5), max_features=400000, stop_words='english', lowercase=True, tokenizer=word_tokenize)),
     ('tfidf', TfidfTransformer(use_idf=True)),
     ('clf', OneVsRestClassifier(LogisticRegression(multi_class='ovr', solver='sag', n_jobs=1, max_iter=100, verbose=0), n_jobs=8))
])
    
print(text_clf.get_params().keys())


# Run Classifier
x_train, y_train = parse_dataset('/home/ichalkidis/nomothesia.nlp.training/nomothesia_nlp/data/datasets/eurovoc_en/train_v3/*.json')

print(x_train.shape)
print(y_train.shape)

text_clf.fit(x_train, y_train)

# Evaluation Report
print('VALIDATION EVALUATION')
print('=====================')
x_val, y_val = parse_dataset('/home/ichalkidis/nomothesia.nlp.training/nomothesia_nlp/data/datasets/eurovoc_en/dev_v3/*.json')
print(x_val.shape)
print(y_val.shape)
calculate_performance(text_clf, x_val, y_val)


print('TEST EVALUATION')
print('=====================')
x_test, y_test = parse_dataset('/home/ichalkidis/nomothesia.nlp.training/nomothesia_nlp/data/datasets/eurovoc_en/test_v3/*.json')
print(x_test.shape)
print(y_test.shape)
calculate_performance(text_clf, x_test, y_test)



