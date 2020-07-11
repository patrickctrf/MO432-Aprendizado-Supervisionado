import datetime
import random

import numpy as np
from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from sklearn.model_selection import ShuffleSplit, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


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
    # Obtem os dados do arquivo CSV.
    df = read_csv("Bias_correction_ucl.csv")
    # Elimina a coluna Next_Tmin.
    df = df.drop(columns=["Next_Tmin"])
    # Elimina a coluna Date
    df = df.drop(columns=["Date"])
    # Elimina todas as linhas que contenham NAN (valor faltante).
    df = df.dropna(axis=0, how='any')

    # Passando os dados para um dataset numpy
    y_data = df["Next_Tmax"].to_numpy()
    X_data = df.drop(columns="Next_Tmax").to_numpy()

    # Scaling dos dados em X.
    scaler = StandardScaler()
    scaler.fit(X_data)
    X_data_scaled = scaler.transform(X_data)

    # ============LINEAR-REGRESSION=============================================

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = LinearRegression()
    cv_results = \
        cross_validate(estimator=regressor, X=X_data_scaled, y=y_data,
                       cv=shuffle_splitter,
                       scoring={"MSE": make_scorer(mean_squared_error, greater_is_better=False),
                                "MAE": make_scorer(mean_absolute_error, greater_is_better=False)})

    print("\n---------------------LINEAR_REGRESSION---------------------")

    print("\nRMSE para cada repetição: \n", (-cv_results["test_MSE"]) ** (1 / 2))

    print("\n\nRMSE médio: ", ((-cv_results["test_MSE"]) ** (1 / 2)).mean())

    # ============L2-RIDGE-REGRESSION===========================================

    np.random.seed(1234)

    # Gera os parametros de entrada aleatoriamente. Alguns sao uniformes nos
    # EXPOENTES.
    alpha = 10 ** np.linspace(-3, 3, 10)

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'alpha': alpha}

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = Ridge()
    cv_results = \
        RandomizedSearchCV(estimator=regressor, cv=shuffle_splitter,
                           param_distributions=parametros,
                           verbose=1,
                           n_jobs=4,
                           scoring="neg_root_mean_squared_error")

    # Realizamos a busca atraves do treinamento
    cv_results.fit(X_data_scaled, y_data)

    print("\n---------------------LINEAR_REGRESSION_L2-------------------")

    print("\nMelhor conjunto de parâmetros: \n", cv_results.best_estimator_)

    print("\nMelhor error score: \n", -cv_results.best_score_)

    # Deafult do sklearn. Coloquei uma lista de 10 parametros iguais so pra nao dar warning, performance nao eh critico aqui
    alpha = [1.0] * 10

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'alpha': alpha}

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = Ridge()
    cv_results = \
        RandomizedSearchCV(estimator=regressor, cv=shuffle_splitter,
                           param_distributions=parametros,
                           verbose=1,
                           n_jobs=4,
                           scoring="neg_root_mean_squared_error")

    # Realizamos a busca atraves do treinamento
    cv_results.fit(X_data_scaled, y_data)

    print("\nScore RMSE default do sklearn: \n", -cv_results.best_score_)

    # ============L1-LASSO-REGRESSION===========================================

    np.random.seed(1234)

    # Gera os parametros de entrada aleatoriamente. Alguns sao uniformes nos
    # EXPOENTES.
    alpha = 10 ** np.linspace(-3, 3, 10)

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'alpha': alpha}

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = Lasso()
    cv_results = \
        RandomizedSearchCV(estimator=regressor, cv=shuffle_splitter,
                           param_distributions=parametros,
                           verbose=1,
                           n_jobs=4,
                           scoring="neg_root_mean_squared_error")

    # Realizamos a busca atraves do treinamento
    cv_results.fit(X_data_scaled, y_data)

    print("\n---------------------LINEAR_REGRESSION_L1-------------------")

    print("\nMelhor conjunto de parâmetros: \n", cv_results.best_estimator_)

    print("\nMelhor error score: \n", -cv_results.best_score_)

    # Deafult do sklearn. Coloquei uma lista de 10 parametros iguais so pra nao dar warning, performance nao eh critico aqui
    alpha = [1.0] * 10

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'alpha': alpha}

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = Lasso()
    cv_results = \
        RandomizedSearchCV(estimator=regressor, cv=shuffle_splitter,
                           param_distributions=parametros,
                           verbose=1,
                           n_jobs=4,
                           scoring="neg_root_mean_squared_error")

    # Realizamos a busca atraves do treinamento
    cv_results.fit(X_data_scaled, y_data)

    print("\nScore RMSE default do sklearn: \n", -cv_results.best_score_)

    # ============SVR-SVM-LINEAR================================================

    np.random.seed(3333)

    # Gera os parametros de entrada aleatoriamente. Alguns sao uniformes nos
    # EXPOENTES.
    c = 2 ** np.linspace(-5, 15, 10)
    epsilon = np.array(random.choices([0.1, 0.3], k=10))

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'C': c, 'epsilon': epsilon}

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=3333)
    regressor = SVR(max_iter=-1, cache_size=7000, kernel="linear")
    cv_results = \
        RandomizedSearchCV(estimator=regressor, cv=shuffle_splitter,
                           param_distributions=parametros,
                           verbose=1,
                           n_jobs=4,
                           scoring="neg_root_mean_squared_error")

    # Realizamos a busca atraves do treinamento
    cv_results.fit(X_data_scaled, y_data)

    print("\n----------------SVR-SVM-LINEAR----------------")

    print("\nMelhor conjunto de parâmetros: \n", cv_results.best_estimator_)

    print("\nMelhor error score: \n", -cv_results.best_score_)

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = SVR(max_iter=-1, cache_size=7000, kernel="linear")
    cv_results = \
        cross_validate(estimator=regressor, X=X_data_scaled, y=y_data,
                       cv=shuffle_splitter,
                       scoring={"MSE": make_scorer(mean_squared_error, greater_is_better=False),
                                "MAE": make_scorer(mean_absolute_error, greater_is_better=False)})

    print("\nScore RMSE parâmetros default: ", ((-cv_results["test_MSE"]) ** (1 / 2)).mean())

    # ============SVR-SVM-RBF================================================

    np.random.seed(3333)

    # Gera os parametros de entrada aleatoriamente. Alguns sao uniformes nos
    # EXPOENTES.
    c = 2 ** np.linspace(-5, 15, 10)
    gamma = 2 ** np.linspace(-9, 3, 10)
    epsilon = np.array(random.choices([0.1, 0.3], k=10))

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'C': c, 'gamma': gamma, 'epsilon': epsilon}

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=3333)
    regressor = SVR(max_iter=-1, cache_size=7000, kernel="rbf")
    cv_results = \
        RandomizedSearchCV(estimator=regressor, cv=shuffle_splitter,
                           param_distributions=parametros,
                           verbose=1,
                           n_jobs=4,
                           scoring="neg_root_mean_squared_error")

    # Realizamos a busca atraves do treinamento
    cv_results.fit(X_data_scaled, y_data)

    print("\n----------------SVR-SVM-RBF----------------")

    print("\nMelhor conjunto de parâmetros: \n", cv_results.best_estimator_)

    print("\nMelhor error score: \n", -cv_results.best_score_)

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = SVR(max_iter=-1, cache_size=7000, kernel="rbf")
    cv_results = \
        cross_validate(estimator=regressor, X=X_data_scaled, y=y_data,
                       cv=shuffle_splitter,
                       scoring={"MSE": make_scorer(mean_squared_error, greater_is_better=False),
                                "MAE": make_scorer(mean_absolute_error, greater_is_better=False)})

    print("\nScore RMSE parâmetros default: ", ((-cv_results["test_MSE"]) ** (1 / 2)).mean())

    # ============KNeighborsRegressor===========================================

    np.random.seed(1234)

    # Gera os parametros de entrada aleatoriamente.
    n_neighbors = np.linspace(1, 1000, 10).astype("int32")

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'n_neighbors': n_neighbors}

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = KNeighborsRegressor()
    cv_results = \
        RandomizedSearchCV(estimator=regressor, cv=shuffle_splitter,
                           param_distributions=parametros,
                           verbose=1,
                           n_jobs=4,
                           scoring="neg_root_mean_squared_error")

    # Realizamos a busca atraves do treinamento
    cv_results.fit(X_data_scaled, y_data)

    print("\n----------------KNeighborsRegressor----------------")

    print("\nMelhor conjunto de parâmetros: \n", cv_results.best_estimator_)

    print("\nMelhor error score: \n", -cv_results.best_score_)

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = KNeighborsRegressor()
    cv_results = \
        cross_validate(estimator=regressor, X=X_data_scaled, y=y_data,
                       cv=shuffle_splitter,
                       scoring={"MSE": make_scorer(mean_squared_error, greater_is_better=False),
                                "MAE": make_scorer(mean_absolute_error, greater_is_better=False)})

    print("\nScore RMSE parâmetros default: ", ((-cv_results["test_MSE"]) ** (1 / 2)).mean())

    # ============MLPRegressor==================================================

    np.random.seed(1234)

    # Gera os parametros de entrada aleatoriamente.
    hidden_layer_sizes = np.array(range(5, 21, 3))

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'hidden_layer_sizes': hidden_layer_sizes}

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = MLPRegressor()
    cv_results = \
        RandomizedSearchCV(estimator=regressor, cv=shuffle_splitter,
                           param_distributions=parametros,
                           verbose=1,
                           n_jobs=4,
                           scoring="neg_root_mean_squared_error")

    # Realizamos a busca atraves do treinamento
    cv_results.fit(X_data_scaled, y_data)

    print("\n---------------MLPRegressor-------------------")

    print("\nMelhor conjunto de parâmetros: \n", cv_results.best_estimator_)

    print("\nMelhor error score: \n", -cv_results.best_score_)

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = MLPRegressor()
    cv_results = \
        cross_validate(estimator=regressor, X=X_data_scaled, y=y_data,
                       cv=shuffle_splitter,
                       scoring={"MSE": make_scorer(mean_squared_error, greater_is_better=False),
                                "MAE": make_scorer(mean_absolute_error, greater_is_better=False)})

    print("\nScore RMSE parâmetros default: ", ((-cv_results["test_MSE"]) ** (1 / 2)).mean())

    # ============DecisionTreeRegressor=========================================

    np.random.seed(1234)

    # Gera os parametros de entrada aleatoriamente.
    ccp_alpha = np.linspace(0, 0.04, 10)

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'ccp_alpha': ccp_alpha}

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = DecisionTreeRegressor()
    cv_results = \
        RandomizedSearchCV(estimator=regressor, cv=shuffle_splitter,
                           param_distributions=parametros,
                           verbose=1,
                           n_jobs=4,
                           scoring="neg_root_mean_squared_error")

    # Realizamos a busca atraves do treinamento
    cv_results.fit(X_data_scaled, y_data)

    print("\n--------------DecisionTreeRegressor------------------")

    print("\nMelhor conjunto de parâmetros: \n", cv_results.best_estimator_)

    print("\nMelhor error score: \n", -cv_results.best_score_)

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = DecisionTreeRegressor()
    cv_results = \
        cross_validate(estimator=regressor, X=X_data_scaled, y=y_data,
                       cv=shuffle_splitter,
                       scoring={"MSE": make_scorer(mean_squared_error, greater_is_better=False),
                                "MAE": make_scorer(mean_absolute_error, greater_is_better=False)})

    print("\nScore RMSE parâmetros default: ", ((-cv_results["test_MSE"]) ** (1 / 2)).mean())

    # ============RandomForestRegressor=========================================

    np.random.seed(1234)

    # Gera os parametros de entrada aleatoriamente.
    n_estimators = [10, 100, 1000]
    max_features = [5, 10, 22]

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'n_estimators': n_estimators, 'max_features': max_features}

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = RandomForestRegressor()
    cv_results = \
        GridSearchCV(estimator=regressor, cv=shuffle_splitter,
                     param_grid=parametros,
                     verbose=1,
                     n_jobs=1,
                     scoring="neg_root_mean_squared_error")

    # Realizamos a busca atraves do treinamento
    cv_results.fit(X_data_scaled, y_data)

    print("\n--------------RandomForestRegressor------------------")

    print("\nMelhor conjunto de parâmetros: \n", cv_results.best_estimator_)

    print("\nMelhor error score: \n", -cv_results.best_score_)

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = RandomForestRegressor()
    cv_results = \
        cross_validate(estimator=regressor, X=X_data_scaled, y=y_data,
                       cv=shuffle_splitter,
                       scoring={"MSE": make_scorer(mean_squared_error, greater_is_better=False),
                                "MAE": make_scorer(mean_absolute_error, greater_is_better=False)})

    print("\nScore RMSE parâmetros default: ", ((-cv_results["test_MSE"]) ** (1 / 2)).mean())

    # ============GradientBoostingRegressor=====================================

    np.random.seed(1234)

    # Gera os parametros de entrada aleatoriamente.
    n_estimators = np.linspace(5, 100, 10).astype("int32")
    learning_rate = [0.01, 0.3]
    max_depth = [2, 3]

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth}

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = GradientBoostingRegressor()
    cv_results = \
        RandomizedSearchCV(estimator=regressor, cv=shuffle_splitter,
                           param_distributions=parametros,
                           verbose=1,
                           n_jobs=4,
                           scoring="neg_root_mean_squared_error")

    # Realizamos a busca atraves do treinamento
    cv_results.fit(X_data_scaled, y_data)

    print("\n--------------GradientBoostingRegressor------------------")

    print("\nMelhor conjunto de parâmetros: \n", cv_results.best_estimator_)

    print("\nMelhor error score: \n", -cv_results.best_score_)

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = GradientBoostingRegressor()
    cv_results = \
        cross_validate(estimator=regressor, X=X_data_scaled, y=y_data,
                       cv=shuffle_splitter,
                       scoring={"MSE": make_scorer(mean_squared_error, greater_is_better=False),
                                "MAE": make_scorer(mean_absolute_error, greater_is_better=False)})

    print("\nScore RMSE parâmetros default: ", ((-cv_results["test_MSE"]) ** (1 / 2)).mean())
