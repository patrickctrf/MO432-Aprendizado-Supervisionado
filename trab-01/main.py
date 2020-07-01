from matplotlib import pyplot as plt
from numpy import hstack, add
from pandas import read_csv, get_dummies
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    data = read_csv("car-data.csv")

    # ============ITEM-1========================================================
    print(data)
    print("\n\n==============================================\n\n")
    print(get_dummies(data))
    get_dummies(data).to_csv("car-with-dummies.csv")

    # OneHot encoding para converter vriaveis ctegoricas em dummy variables.
    data = get_dummies(data)

    # Separamos dados de entrada dos dados de saida.
    # A coluna a remover ("i") eh a do selling_price.
    i = 1
    X_data = hstack((data.to_numpy()[:, 0:i], data.to_numpy()[:, i + 1:]))
    # X_data = X_data.T
    y_data = data.to_numpy()[:, i]

    # ============ITEM-2========================================================
    scaler = StandardScaler()
    scaler.fit(X_data)

    print("Médias do Scaler em cada coluna: ", scaler.mean_)
    print('Dados centrados e "escalados": ', scaler.transform(X_data))

    X_data_scaled = scaler.transform(X_data)

    # ============ITEM-3========================================================
    pca = PCA(n_components=0.9, svd_solver="full")
    pca.fit(X_data_scaled)

    print("Números de componentes do PCA para 90% de explicação da variância: ", pca.n_components_)

    plt.plot(pca.explained_variance_ratio_)
    plt.title('Taxa de explicação da variância por componente')
    plt.ylabel('Taxa de explicação da Variância')
    plt.xlabel('Número de Componentes')
    plt.show()

    plt.plot(add.accumulate(pca.explained_variance_ratio_))
    plt.title('Explicação da variância acumulada por componente')
    plt.ylabel('Taxa de explicação acumulada da Variância')
    plt.xlabel('Número de Componentes')
    plt.show()

    # Obtido graficamente
    pca = PCA(n_components=15)
    pca.fit(X_data_scaled)
    X_data_pca = pca.transform(X_data_scaled)

    # ============ITEM-4========================================================
    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=0.3, random_state=1234)
    regressor = LinearRegression()
    cv_results = cross_validate(regressor, X_data_pca, y_data,
                                cv=shuffle_splitter,
                                scoring={"MSE": make_scorer(mean_squared_error, greater_is_better=False),
                                         "MAE": make_scorer(mean_absolute_error, greater_is_better=False)})

    # print(cv_results.keys())
    print("\nRMSE para cada repetição: \n", (-cv_results["test_MSE"]) ** (1 / 2))
    print("\n\nMAE para cada repetição: \n", -cv_results["test_MAE"])

    print("\n\nRMSE médio: ", ((-cv_results["test_MSE"]) ** (1 / 2)).mean())
    print("\nMAE médio: ", (-cv_results["test_MAE"]).mean())
