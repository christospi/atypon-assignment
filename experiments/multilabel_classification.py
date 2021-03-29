import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

from classifiers.svm_model import SVMClassifier


class MultilabelClassification:
    
    def __init__(self) -> None:
        # self.dataset = 'data/dataset/small_assignment_dataset.json'
        self.dataset = 'data/dataset/assignment_dataset.json'
        self.samples = None
        self.targets = None
        self.targets_dict = dict()

    def load_dataset_to_df(self):
        mlb = MultiLabelBinarizer()
        data = pd.read_json(self.dataset)
        self.samples = data['text']
        self.targets = data['labels']
        self.targets = pd.DataFrame(mlb.fit_transform(self.targets), columns=mlb.classes_, index=self.targets.index)
        for i, col in enumerate(self.targets.columns):
            self.targets_dict[i] = col

    def print_data_stats(self, prefix='full_data_'):
        # generate plot
        self.targets.sum().plot(kind='barh', figsize=(10, 10), color='steelblue')
        plt.xlabel('Number of Articles')
        plt.title('Labels occurrence in dataset')

        # annotate value labels to each country
        for index, value in enumerate(self.targets.sum()):
            label = format(int(value), ',')  # format int with commas

            # place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)
            plt.annotate(label, xy=(value - 180, index - 0.15), color='white')

        plt.savefig('resources/{}barh.png'.format(prefix))
        plt.show()

    def train_classifier(self, model='svm'):
        if model == 'svm':
            classifier = SVMClassifier()
            classifier.train(self.samples, self.targets)
