import datetime
import random

import numpy as np
from pandas import read_csv, get_dummies
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.model_selection import ShuffleSplit, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
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
    df = read_csv("dados3.csv")
    # # Elimina a coluna Next_Tmin.
    # df = df.drop(columns=["Next_Tmin"])
    # # Elimina a coluna Date
    # df = df.drop(columns=["Date"])
    # Elimina todas as linhas que contenham NAN (valor faltante).
    df = df.dropna(axis=0, how='any')

    get_dummies(df).to_csv("dados3-dummies.csv")

    # OneHot encoding para converter vriaveis ctegoricas em dummy variables.
    df = get_dummies(df)

    # Passando os dados para um dataset numpy
    y_data = df["V15"].to_numpy()
    X_data = df.drop(columns="V15").to_numpy()

    # Scaling dos dados em X.
    scaler = StandardScaler()
    scaler.fit(X_data)
    X_data_scaled = scaler.transform(X_data)

    # ============Logistic-Regression===========================================
    np.random.seed(1234)

    print("\n----------------Logistic-Regression----------------")

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = LogisticRegression(penalty=None)
    cv_results = \
        cross_validate(estimator=regressor, X=X_data_scaled, y=y_data,
                       cv=shuffle_splitter,
                       scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})

    print("\nScore AUC parâmetros default: ", (cv_results["test_AUC"]).mean())

    # ============Logistic-Regression-L2========================================
    np.random.seed(3333)

    # Gera os parametros de entrada aleatoriamente. Alguns sao uniformes nos
    # EXPOENTES.
    c = 10 ** np.random.uniform(-3, 3, 10)

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'C': c}

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=3333)
    regressor = LogisticRegression()
    cv_results = \
        RandomizedSearchCV(estimator=regressor, cv=shuffle_splitter,
                           param_distributions=parametros,
                           refit="AUC",
                           verbose=1,
                           n_jobs=4,
                           scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})

    # Realizamos a busca atraves do treinamento
    cv_results.fit(X_data_scaled, y_data)

    print("\n----------------Logistic-Regression-L2----------------")

    print("\nMelhor conjunto de parâmetros: \n", cv_results.best_estimator_)

    print("\nMelhor error score: \n", -cv_results.best_score_)

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = LogisticRegression()
    cv_results = \
        cross_validate(estimator=regressor, X=X_data_scaled, y=y_data,
                       cv=shuffle_splitter,
                       scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})

    print("\nScore AUC parâmetros default: ", (cv_results["test_AUC"]).mean())

    # ============LDA===========================================================
    np.random.seed(1234)

    print("\n----------------LDA----------------")

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = LinearDiscriminantAnalysis()
    cv_results = \
        cross_validate(estimator=regressor, X=X_data_scaled, y=y_data,
                       cv=shuffle_splitter,
                       scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})

    print("\nScore AUC parâmetros default: ", (cv_results["test_AUC"]).mean())

    # ============QDA===========================================================
    np.random.seed(1234)

    print("\n----------------QDA----------------")

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = QuadraticDiscriminantAnalysis()
    cv_results = \
        cross_validate(estimator=regressor, X=X_data_scaled, y=y_data,
                       cv=shuffle_splitter,
                       scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})

    print("\nScore AUC parâmetros default: ", (cv_results["test_AUC"]).mean())

    # # ============SVC-SVM-LINEAR================================================
    # np.random.seed(3333)
    #
    # # Gera os parametros de entrada aleatoriamente. Alguns sao uniformes nos
    # # EXPOENTES.
    # c = 2 ** np.random.uniform(-5, 15, 10)
    #
    # # Une os parametros de entrada em um unico dicionario a ser passado para a
    # # funcao.
    # parametros = {'C': c}
    #
    # shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=3333)
    # regressor = SVC(max_iter=-1, cache_size=7000, kernel="linear", probability=True)
    # cv_results = \
    #     RandomizedSearchCV(estimator=regressor, cv=shuffle_splitter,
    #                        param_distributions=parametros,
    #                        refit="AUC",
    #                        verbose=1,
    #                        n_jobs=4,
    #                        scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})
    #
    # # Realizamos a busca atraves do treinamento
    # cv_results.fit(X_data_scaled, y_data)
    #
    # print("\n----------------SVC-SVM-LINEAR----------------")
    #
    # print("\nMelhor conjunto de parâmetros: \n", cv_results.best_estimator_)
    #
    # print("\nMelhor error score: \n", -cv_results.best_score_)
    #
    # shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    # regressor = SVC(max_iter=-1, cache_size=7000, kernel="linear", probability=True)
    # cv_results = \
    #     cross_validate(estimator=regressor, X=X_data_scaled, y=y_data,
    #                    cv=shuffle_splitter,
    #                    scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})
    #
    # print("\nScore AUC parâmetros default: ", (cv_results["test_AUC"]).mean())
    #
    # # ============SVC-SVM-RBF===================================================
    # np.random.seed(3333)
    #
    # # Gera os parametros de entrada aleatoriamente. Alguns sao uniformes nos
    # # EXPOENTES.
    # c = 2 ** np.random.uniform(-5, 15, 10)
    # gamma = 2 ** np.random.uniform(-9, 3, 10)
    #
    # # Une os parametros de entrada em um unico dicionario a ser passado para a
    # # funcao.
    # parametros = {'C': c, 'gamma': gamma}
    #
    # shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=3333)
    # regressor = SVC(max_iter=-1, cache_size=7000, kernel="rbf", probability=True)
    # cv_results = \
    #     RandomizedSearchCV(estimator=regressor, cv=shuffle_splitter,
    #                        param_distributions=parametros,
    #                        refit="AUC",
    #                        verbose=1,
    #                        n_jobs=4,
    #                        scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})
    #
    # # Realizamos a busca atraves do treinamento
    # cv_results.fit(X_data_scaled, y_data)
    #
    # print("\n----------------SVC-SVM-RBF----------------")
    #
    # print("\nMelhor conjunto de parâmetros: \n", cv_results.best_estimator_)
    #
    # print("\nMelhor error score: \n", -cv_results.best_score_)
    #
    # shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    # regressor = SVC(max_iter=-1, cache_size=7000, kernel="rbf", probability=True)
    # cv_results = \
    #     cross_validate(estimator=regressor, X=X_data_scaled, y=y_data,
    #                    cv=shuffle_splitter,
    #                    scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})
    #
    # print("\nScore AUC parâmetros default: ", (cv_results["test_AUC"]).mean())

    # ============KNeighborsRegressor===========================================
    np.random.seed(1234)

    # Gera os parametros de entrada aleatoriamente.
    n_neighbors = np.random.uniform(0, 150, 10).astype("int32") * 2 + 1

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'n_neighbors': n_neighbors}

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = KNeighborsClassifier()
    cv_results = \
        RandomizedSearchCV(estimator=regressor, cv=shuffle_splitter,
                           param_distributions=parametros,
                           refit="AUC",
                           verbose=1,
                           n_jobs=4,
                           scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})

    # Realizamos a busca atraves do treinamento
    cv_results.fit(X_data_scaled, y_data)

    print("\n----------------KNeighborsClassifier----------------")

    print("\nMelhor conjunto de parâmetros: \n", cv_results.best_estimator_)

    print("\nMelhor error score: \n", -cv_results.best_score_)

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = KNeighborsClassifier()
    cv_results = \
        cross_validate(estimator=regressor, X=X_data_scaled, y=y_data,
                       cv=shuffle_splitter,
                       scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})

    print("\nScore AUC parâmetros default: ", (cv_results["test_AUC"]).mean())

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
                           refit="AUC",
                           verbose=1,
                           n_jobs=4,
                           scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})

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
                       scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})

    print("\nScore AUC parâmetros default: ", (cv_results["test_AUC"]).mean())

    # ============DecisionTreeRegressor=========================================

    np.random.seed(1234)

    # Gera os parametros de entrada aleatoriamente.
    ccp_alpha = np.random.uniform(0, 0.04, 10)

    # Une os parametros de entrada em um unico dicionario a ser passado para a
    # funcao.
    parametros = {'ccp_alpha': ccp_alpha}

    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = DecisionTreeRegressor()
    cv_results = \
        RandomizedSearchCV(estimator=regressor, cv=shuffle_splitter,
                           param_distributions=parametros,
                           refit="AUC",
                           verbose=1,
                           n_jobs=4,
                           scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})

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
                       scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})

    print("\nScore AUC parâmetros default: ", (cv_results["test_AUC"]).mean())

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
                     scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})

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
                       scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})

    print("\nScore AUC parâmetros default: ", (cv_results["test_AUC"]).mean())

    # ============GradientBoostingRegressor=====================================

    np.random.seed(1234)

    # Gera os parametros de entrada aleatoriamente.
    n_estimators = np.random.uniform(5, 100, 10).astype("int32")
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
                           refit="AUC",
                           verbose=1,
                           n_jobs=4,
                           scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})

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
                       scoring={"AUC": make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)})

    print("\nScore AUC parâmetros default: ", (cv_results["test_AUC"]).mean())
