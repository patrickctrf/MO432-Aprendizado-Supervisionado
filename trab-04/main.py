import datetime

from pandas import read_csv
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
    data = df["Taxa"].to_numpy()
    data = data.reshape(-1, 1)
    train_data = data[0:int(data.size * 0.9)]
    validation_data = data[int(data.size * 0.9):]

    # Scaling dos dados. Importante fazer o FIT so nos dados de treino (senao eh contaminacao)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    validation_data = scaler.transform(validation_data)

    x = 0
