from experiments.multilabel_classification import MultilabelClassification


def run():
    mltc = MultilabelClassification()
    mltc.load_dataset_to_df()
    # mltc.print_data_stats()
    mltc.train_classifier()


if __name__ == '__main__':
    run()
