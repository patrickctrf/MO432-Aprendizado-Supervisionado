from numpy import ones, where, array, array_split


def time_series_split(data_x, data_y=None, sampling_windows_size=10, n_steps_prediction=1, stateful=False, is_classifier=False, threshold=0):
    """
Split the given time series into input (X) and observed (y) data.
There are 3 principal modes of splitting data with this function: Stateful single series, non stateful single series, and non stateful multi-series.

When you have the input data and the output data inside the same array, we consider it as a single series and all data is passed through the data_x parameters.
If the input data does not contain your observed information (y), the input array is given in the data_x parameter and the prediction target goes to the data_y parameter.

It's also available a stateful splitting mode if you are working with a stateful LSTM, for example.

There is no stateful multi-series option because, in this case, your input array or matrix (X) and your observed array or matrix data (y) would be already given to you by definition.

    :param data_x: array-like input data from your series.
    :param data_y: array-like observed data (target) for your series. Let it be 'None' if dealing with a single component series.
    :param sampling_windows_size: Size of window sampling (W) or time steps entering your network for each prediction. Must be positive integer.
    :param n_steps_prediction: How many steps it is going to predict ahead. Must be positive integer.
    :param stateful: True or False, indicating whether your network are suposedto work statefully or not, respectively.
    :param is_classifier: If True, the 'y' output data is transformed into +1 or -1, according to 'threshold' selected by user. Useful for non quantitative prediction (classification, not regression).
    :param threshold: Threshold for 'is_classifier' parameter. Float in the interval of your observed data 'data_y'.
    :return: X input and y observed data, formatted according to the given function parameters.
    """
    if data_y is None:
        data_y = data_x

    # Converte os dados para array numpy, para podermos utilizar a API dos arrays.
    data_x = array(data_x)
    data_y = array(data_y)

    if stateful is False:
        # arrays para armazenar as saidas
        X = ones((data_x.shape[0] - n_steps_prediction - sampling_windows_size, sampling_windows_size) + data_x.shape[1:])
        y = ones((data_y.shape[0] - n_steps_prediction - sampling_windows_size, n_steps_prediction) + data_y.shape[1:])

        i = 0
        # Precisamos de N amostras (sampling_windows_size) e as amostras de
        # ground truth (n_steps_prediction) para fazer mais um split.
        while i < data_x.shape[0] - n_steps_prediction - sampling_windows_size:
            X[i] = (data_x[i:i + sampling_windows_size])
            y[i] = (data_y[i + sampling_windows_size:(i + sampling_windows_size) + n_steps_prediction])

            i += 1

    # Se estivermos trabalhando com uma serie stateful, a unica demanda eh
    # deslocar a entrada e a saida em uma unidade (a entrada eh tudo o que vimos
    # ate agora e a saida eh o proximo step, que sera conhecido no passo seguinte).
    else:
        X = data_x[:-1]
        y = data_y[1:]

    if is_classifier is True:
        y = where(y > threshold, 1.0, -1.0)

    return X, y


class TimeSeriesSplitCV():
    def __init__(self, n_splits=5, training_percent=0.7, sampling_windows_size=10, n_steps_prediction=1, stateful=False, is_classifier=False, threshold=0):
        """
Time series split with cross validation separation as a compatible sklearn-like splitter.
There are 3 principal modes of splitting data with this function: Stateful single series, non stateful single series, and non stateful multi-series.

When you have the input data and the output data inside the same array, we consider it as a single series and all data is passed through the data_x parameters.
If the input data does not contain your observed information (y), the input array is given in the data_x parameter and the prediction target goes to the data_y parameter.

It's also available a stateful splitting mode if you are working with a stateful LSTM, for example.

There is no stateful multi-series option because, in this case, your input array or matrix (X) and your observed array or matrix data (y) would be already given to you by definition.


        :param n_splits: Like k-folds split, how many sub series to split.
        :param training_percent: Ratio between train and validation data for cross validation.
        :param sampling_windows_size: Size of window sampling (W) or time steps entering your network for each prediction. Must be positive integer.
        :param n_steps_prediction: How many steps it is going to predict ahead. Must be positive integer.
        :param stateful: True or False, indicating whether your network are suposedto work statefully or not, respectively.
        :param is_classifier: If True, the 'y' output data is transformed into +1 or -1, according to 'threshold' selected by user. Useful for non quantitative prediction (classification, not regression).
        :param threshold: Threshold for 'is_classifier' parameter. Float in the interval of your observed data 'data_y'.
        :return: X input and y observed data, formatted according to the given function parameters.
        """
        self.n_splits = n_splits
        self.training_percent = training_percent
        self.sampling_windows_size = sampling_windows_size
        self.n_steps_prediction = n_steps_prediction
        self.stateful = stateful
        self.is_classifier = is_classifier
        self.threshold = threshold

    def split(self, X, y=None, groups=None):
        """
Generate indices to split data into training and test set.

        :param X: array-like input data from your series.
        :param y: array-like observed data (target) for your series. Let it be 'None' if dealing with a single component series.
        :param groups: Always ignored, exists for compatibility.
        """
        X = array(X)
        y = array(y)

        splits_indices = array_split(range(X.shape[0]), self.n_splits)

        train_list = []
        test_list = []

        for i, single_split in enumerate(splits_indices):
            train = single_split[:int(single_split.shape[0]*self.training_percent)]
            test = single_split[int(single_split.shape[0] * self.training_percent):]

            train_list.append(train)
            test_list.append(test)

        return train_list, test_list


