import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from classifiers.rnn_model import RNNClassifier
from classifiers.log_reg_model import LogRegClassifier
from configuration import config


class MultilabelClassification:
    
    def __init__(self) -> None:
        self.dataset = config.props['dataset']
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

    def train_classifier(self):
        if config.props['model'] == 'logreg':
            classifier = LogRegClassifier()
        elif config.props['model'] == 'bilstm':
            classifier = RNNClassifier()
        else:
            raise Exception('Requested model not supported. Available: ["svm", "bilstm"]')

        classifier.train(self.samples, self.targets)

    def print_data_stats(self, prefix='full_data_'):

        to_plot = False

        # Plot 1
        # Plot labels frequency/occurrence in articles
        if to_plot:
            self.targets.sum().sort_values().plot(kind='barh', figsize=(10, 10), color='steelblue')
            plt.xlabel('Number of Articles')
            plt.title('Labels occurrence in dataset')

            # annotate value labels to each country
            for index, value in enumerate(self.targets.sum().sort_values()):
                label = format(int(value), ',')  # format int with commas

                # place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)
                plt.annotate(label, xy=(value - 180, index - 0.15), color='white')

            plt.savefig('resources/{}barh.png'.format(prefix))
            plt.show()

        # Plot 2
        # Plot tokens distribution to articles
        if to_plot:
            print('='*32, '\n', 'TOKENS STATS')
            print('='*32)
            lengths = []
            for article in self.samples:
                lengths.append(len(article.split()))

            print('TOTAL: AVERAGE: {:.2f} MIN: {} MAX: {}'.format(
                sum(lengths)/len(self.samples), min(lengths), max(lengths))
            )

            lengths = sorted(lengths)
            for i in range(20):
                end = int(((i + 1) / 10) * len(lengths) / 2)
                print(
                    'DATASET ({0:3}%): ARTICLES: {1:5} LENGTH: MAX: {2:3} MEAN: {3:.2f}'.format(
                        (i + 1) * 5, end, max(lengths[:end]), np.mean(lengths[:end])))

            plt.hist(lengths, bins=20, range=(0, 300))
            plt.axvline(np.array(lengths).mean(), color='k', linestyle='dashed', linewidth=1,
                        label='Mean: {}'.format(int(np.array(lengths).mean())))
            plt.legend(loc='upper right')
            plt.title('Tokens Distribution')
            plt.ylabel('Number of Articles')
            plt.xlabel("Number of Tokens")
            plt.savefig('resources/{}tokens_stats.png'.format(prefix))
            plt.show()
