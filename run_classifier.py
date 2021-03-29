from experiments.multilabel_classification import MultilabelClassification


def run(model='svm'):
    mltc = MultilabelClassification()
    mltc.load_dataset_to_df()
    mltc.print_data_stats()

    mltc.train_classifier(model)


if __name__ == '__main__':
    run(model='svm')
