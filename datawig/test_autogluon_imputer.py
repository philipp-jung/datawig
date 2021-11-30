import pytest
import random
import numpy as np
import pandas as pd
from datawig import AutoGluonImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score


def rand_string(length: int = 16) -> str:
    """
    Utility function for generating a random alphanumeric string of specified length

    :param length: length of the generated string

    :return: random string
    """
    import string
    return ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(length)])


@pytest.fixture
def data_frame():

    def _inner_impl(
            feature_col='features',
            label_col='labels',
            n_samples=500,
            word_length=5,
            num_words=100,
            vocab_size=100,
            num_labels=10):

        """
        Generates text features and categorical labels.
        :param feature_col: name of feature column.
        :param label_col: name of label column.
        :param n_samples: how many rows to generate.
        :return: pd.DataFrame with columns = [feature_col, label_col]
        """

        vocab = [rand_string(word_length) for _ in range(vocab_size)]
        labels = vocab[:num_labels]
        words = vocab[num_labels:]

        def _sentence_with_label(labels=labels, words=words):
            """
            Generates a random token sequence containing a random label

            :param labels: label set
            :param words: vocabulary of tokens
            :return: blank separated token sequence and label

            """
            label = random.choice(labels)
            tokens = [random.choice(words) for _ in range(num_words)] + [label]
            sentence = " ".join(np.random.permutation(tokens))

            return sentence, label

        sentences, labels = zip(*[_sentence_with_label(labels, words) for _ in range(n_samples)])
        df = pd.DataFrame({feature_col: sentences, label_col: labels})

        return df

    return _inner_impl


def test_quantile_regression(data_frame):
    """
    Basic integration test that tests AutoGluonImputer's quantile regression
    with default settings.
    """

    feature_col = "string_feature"
    label_col = "label"

    n_samples = 1000
    num_labels = 3
    seq_len = 30
    vocab_size = int(2 ** 10)

    # generate some random data
    random_data = data_frame(feature_col=feature_col,
                             label_col=label_col,
                             vocab_size=vocab_size,
                             num_labels=num_labels,
                             num_words=seq_len,
                             n_samples=n_samples)

    numeric_data = np.random.uniform(-np.pi, np.pi, (n_samples,))
    df = pd.DataFrame({
        'x': numeric_data,
        '*2': numeric_data * 2. + np.random.normal(0, .1, (n_samples,)),
        '**2': numeric_data ** 2 + np.random.normal(0, .1, (n_samples,)),
        feature_col: random_data[feature_col].values,
        label_col: random_data[label_col].values
    })

    df_train, df_test = train_test_split(df, test_size=.1)

    imputer = AutoGluonImputer(
        model_name="test_quadratic",
        input_columns=['x', feature_col],
        output_column="*2",
        precision_threshold=.01,
        numerical_confidence_quantile=.01
    )

    imputer.fit(train_df=df_train, time_limit=10)
    df_pred = imputer.predict(df_test)

    assert mean_squared_error(df_test['*2'], df_pred['*2_imputed']) < 1.0


def test_precision_threshold(data_frame):
    """
    Basic integration test that tests AutoGluonImputer's ability to impute categorical
    features with default settings.
    If this fails, some functionality around the approach used to derive the
    precision thresholds is defunct.
    """

    feature_col = "string_feature"
    label_col = "label"

    n_samples = 1000
    num_labels = 3
    seq_len = 30
    vocab_size = int(2 ** 10)

    # generate some random data
    random_data = data_frame(feature_col=feature_col,
                             label_col=label_col,
                             vocab_size=vocab_size,
                             num_labels=num_labels,
                             num_words=seq_len,
                             n_samples=n_samples)

    numeric_data = np.random.uniform(-np.pi, np.pi, (n_samples,))
    df = pd.DataFrame({
        'x': numeric_data,
        '*2': numeric_data * 2. + np.random.normal(0, .1, (n_samples,)),
        '**2': numeric_data ** 2 + np.random.normal(0, .1, (n_samples,)),
        feature_col: random_data[feature_col].values,
        label_col: random_data[label_col].values
    })

    df_train, df_test = train_test_split(df, test_size=.1)

    imputer = AutoGluonImputer(
        model_name="test_quadratic",
        input_columns=['x', feature_col],
        output_column="label",
        precision_threshold=.01,
        numerical_confidence_quantile=.01
    )

    imputer.fit(train_df=df_train, time_limit=10)
    df_pred = imputer.predict(df_test)

    assert f1_score(df_test['label'], df_pred['label_imputed']) > 0.7
