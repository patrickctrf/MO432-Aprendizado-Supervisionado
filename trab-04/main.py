import datetime

from numpy import true_divide, hstack, ones
from numpy.random import uniform, seed
from pandas import read_csv
from ptk.timeseries import time_series_split, TimeSeriesSplitCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import MinMaxScaler


def date_to_timestamp(date_string):
    """
Receives a string containing a datetime and returns an equivalent timestamp

    :param date_string: Date stringto become timestamp
    :return: Timestamp (float)
    """
    element = datetime.datetime.strptime(date_string, "%Y-%m-%d")

    timestamp = datetime.datetime.timestamp(element)

    return timestamp


if __name__ == '__main__':
    seed(1234)

    # Obtem os dados do arquivo CSV.
    df = read_csv("dados4.csv")
    # Elimina a coluna Date
    df = df.drop(columns=["Data"])
    # Elimina todas as linhas que contenham NAN (valor faltante).
    df = df.dropna(axis=0, how='any')
    # Elimina todas as linhas que contenham "-" (que eh um valor faltante).
    df = df[df.Taxa != "-"]

    df.to_csv("dados4-tratado.csv")

    # get_dummies(df).to_csv("dados4-dummies.csv")
    # OneHot encoding para converter vriaveis ctegoricas em dummy variables.
    # df = get_dummies(df)

    # Passando os dados para um dataset numpy
    # y_data = df["V15"].to_numpy()
    # X_data = df.drop(columns="V15").to_numpy()

    # Separando o dataset para nested cross validation.
    data = df["Taxa"].to_numpy()[::-1].astype("float")
    data = data.reshape(-1, 1)
    train_data_x = data[0:int(data.size * 0.9)]
    validation_data_x = data[int(data.size * 0.9):]
    train_data_y = data[0:int(data.size * 0.9)]
    validation_data_y = data[int(data.size * 0.9):]

    # Scaling dos dados. Importante fazer o FIT so nos dados de treino (senao eh contaminacao)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_data_x)
    train_data_x = scaler.transform(train_data_x)
    validation_data_x = scaler.transform(validation_data_x)

    # X, y = time_series_split(train_data.reshape(-1), sampling_window_size=10, n_steps_prediction=1, is_classifier=False)

    # splitter_object = TimeSeriesSplitCV(sampling_window_size=10, n_splits=5, training_percent=0.7, blocking_split=False)
    # train, test = splitter_object.split(X, y)

    # =========Cross-Validacao-dos-dados-em-diferentes-regresssor===============
    #
    # regressor_list = [GradientBoostingRegressor(),
    #                   RandomForestRegressor(max_features=5, n_estimators=1000),
    #                   MLPRegressor(hidden_layer_sizes=20)
    #                   ]
    #
    # regressor_names = ["GradientBoostingRegressor()",
    #                    "RandomForestRegressor(max_features=5, n_estimators=1000)",
    #                    "MLPRegressor(hidden_layer_sizes=20)"
    #                    ]
    #
    # w_list = uniform(6, 20, 5).astype("int32")
    #
    # with open("regressores.txt", "w") as log_file:
    #
    #     for w in w_list:
    #         X, y = time_series_split(train_data_x.reshape(-1), train_data_y.reshape(-1), sampling_window_size=w, n_steps_prediction=1, is_classifier=False)
    #         for regressor, name in zip(regressor_list, regressor_names):
    #             splitter_cv = TimeSeriesSplitCV(sampling_window_size=10, n_splits=5,
    #                                             training_percent=0.7,
    #                                             blocking_split=False)
    #             cv_results = \
    #                 cross_validate(estimator=regressor, X=X, y=y,
    #                                cv=splitter_cv, n_jobs=4,
    #                                scoring={"MSE": make_scorer(mean_squared_error, greater_is_better=False),
    #                                         "MAE": make_scorer(mean_absolute_error, greater_is_better=False)})
    #
    #             log_file.write("\nScore RMSE " + name + " : " + str(((-cv_results["test_MSE"]) ** (1 / 2)).mean()) + "\nW size: " + str(w))
    #             print("\nScore RMSE " + name + " : " + str(((-cv_results["test_MSE"]) ** (1 / 2)).mean()) + "\nW size: " + str(w))
    #
    # # =========Cross-Validacao-dos-dados-em-diferentes-classificadores==========
    #
    # regressor_list = [GradientBoostingClassifier(),
    #                   RandomForestClassifier(max_features=5, n_estimators=1000),
    #                   MLPClassifier(hidden_layer_sizes=20),
    #                   LogisticRegression()
    #                   ]
    #
    # regressor_names = ["GradientBoostingClassifier()",
    #                    "RandomForestClassifier(max_features=5, n_estimators=1000)",
    #                    "MLPClassifier(hidden_layer_sizes=20)",
    #                    "LogisticRegression()"
    #                    ]
    #
    # w_list = uniform(8, 20, 5).astype("int32")
    #
    # with open("classificadores.txt", "w") as log_file:
    #
    #     for w in w_list:
    #         X, y = time_series_split(data_x=train_data_x.reshape(-1),
    #                                  data_y=hstack((ones((1,)), true_divide(train_data_y[1:].reshape(-1), train_data_y[:-1].reshape(-1)))),
    #                                  sampling_window_size=w,
    #                                  n_steps_prediction=1,
    #                                  is_classifier=True,
    #                                  threshold=1)
    #         for regressor, name in zip(regressor_list, regressor_names):
    #             splitter_cv = TimeSeriesSplitCV(sampling_window_size=10, n_splits=5,
    #                                             training_percent=0.7,
    #                                             blocking_split=False)
    #             cv_results = \
    #                 cross_validate(estimator=regressor, X=X, y=y,
    #                                cv=splitter_cv, n_jobs=4,
    #                                scoring={"ACC": "accuracy"})
    #
    #             log_file.write("\nScore acurácia " + name + " : " + str((cv_results["test_ACC"]).mean()) + "\nW size: " + str(w))
    #             print("\nScore acurácia " + name + " : " + str((cv_results["test_ACC"]).mean()) + "\nW size: " + str(w))

    # =========Melhores-Estimadores-Agora-No-Conjunto-de_Medida=================

    # Regressao
    X, y = time_series_split(train_data_x.reshape(-1), train_data_y.reshape(-1), sampling_window_size=8, n_steps_prediction=1, is_classifier=False)
    regressor = RandomForestRegressor(max_features=5, n_estimators=1000)
    regressor.fit(X, y)

    X, y = time_series_split(validation_data_x.reshape(-1), validation_data_y.reshape(-1), sampling_window_size=8, n_steps_prediction=1, is_classifier=False)
    yhat = regressor.predict(X)

    print("RMSE do random forest no conjunto medida: ", mean_squared_error(yhat, y) ** 0.5)

    # Classificacao
    X, y = time_series_split(data_x=train_data_x.reshape(-1),
                             data_y=hstack((ones((1,)), true_divide(train_data_y[1:].reshape(-1), train_data_y[:-1].reshape(-1)))),
                             sampling_window_size=17,
                             n_steps_prediction=1,
                             is_classifier=True,
                             threshold=1)

    classificador = RandomForestClassifier(max_features=5, n_estimators=1000)
    classificador.fit(X, y)

    X, y = time_series_split(data_x=validation_data_x.reshape(-1),
                             data_y=hstack((ones((1,)), true_divide(validation_data_y[1:].reshape(-1), validation_data_y[:-1].reshape(-1)))),
                             sampling_window_size=17,
                             n_steps_prediction=1,
                             is_classifier=True,
                             threshold=1)
    yhat = classificador.predict(X)

    print("Acurácia do random forest no conjunto medida: ", accuracy_score(yhat, y))

    # X, y = time_series_split(data_x=train_data_x.reshape(-1),
    #                          data_y=hstack((ones((1,)), true_divide(train_data_y[1:].reshape(-1), train_data_y[:-1].reshape(-1)))),
    #                          sampling_window_size=17,
    #                          n_steps_prediction=1,
    #                          is_classifier=True,
    #                          threshold=1)
    #
    # splitter_cv = TimeSeriesSplitCV(sampling_window_size=17, n_splits=5,
    #                                 training_percent=0.7,
    #                                 blocking_split=False)
    # cv_results = \
    #     cross_validate(estimator=RandomForestClassifier(max_features=5, n_estimators=1000), X=X, y=y,
    #                    cv=splitter_cv, n_jobs=4, return_estimator=True,
    #                    scoring={"ACC": "accuracy"})
    #
    # print(cv_results)
    #
    # # Classificacao
    # X, y = time_series_split(data_x=train_data_x.reshape(-1),
    #                          data_y=hstack((ones((1,)), true_divide(train_data_y[1:].reshape(-1), train_data_y[:-1].reshape(-1)))),
    #                          sampling_window_size=17,
    #                          n_steps_prediction=1,
    #                          is_classifier=True,
    #                          threshold=1)
    #
    # classificador = cv_results["estimator"][-1]
    #
    # X, y = time_series_split(data_x=validation_data_x.reshape(-1),
    #                          data_y=hstack((ones((1,)), true_divide(validation_data_y[1:].reshape(-1), validation_data_y[:-1].reshape(-1)))),
    #                          sampling_window_size=17,
    #                          n_steps_prediction=1,
    #                          is_classifier=True,
    #                          threshold=1)
    # yhat = classificador.predict(X)
    #
    # print("Acurácia do random forest no conjunto medida: ", accuracy_score(yhat, y))