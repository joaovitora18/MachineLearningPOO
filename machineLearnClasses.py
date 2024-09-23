import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

class DataLoader:
    """
    Classe para carregar dados de um arquivo Excel.

    Atributos:
    ----------
    df : pandas.DataFrame
        DataFrame contendo os dados carregados do arquivo Excel.

    Métodos:
    --------
    get_data():
        Retorna o DataFrame com os dados carregados.
    """

    def __init__(self, file_path):
        """
        Inicializa a classe DataLoader com o caminho do arquivo.

        Parâmetros:
        -----------
        file_path : str
            Caminho para o arquivo Excel.
        """
        self.df = pd.read_excel(file_path)

    def get_data(self):
        """
        Retorna o DataFrame com os dados carregados.

        Retorna:
        --------
        pandas.DataFrame
            DataFrame contendo os dados carregados.
        """
        return self.df


class DataPreprocessor:
    """
    Classe para pré-processamento de dados.

    Atributos:
    ----------
    df : pandas.DataFrame
        DataFrame contendo os dados a serem pré-processados.
    le_nome : LabelEncoder
        Codificador para a coluna 'Nome'.
    le_cor : LabelEncoder
        Codificador para a coluna 'cor preferida'.

    Métodos:
    --------
    preprocess():
        Realiza o pré-processamento dos dados e retorna o DataFrame processado.
    get_encoders():
        Retorna os codificadores de 'Nome' e 'cor preferida'.
    """

    def __init__(self, df):
        """
        Inicializa a classe DataPreprocessor com um DataFrame.

        Parâmetros:
        -----------
        df : pandas.DataFrame
            DataFrame contendo os dados a serem pré-processados.
        """
        self.df = df
        self.le_nome = LabelEncoder()
        self.le_cor = LabelEncoder()

    def preprocess(self):
        """
        Realiza o pré-processamento dos dados.

        - Remove a coluna 'IP Address'.
        - Converte a coluna 'Added Time' para o formato '%d/%m/%y - %H:%M:%S'.
        - Ordena os dados pela coluna 'Nome'.
        - Codifica as colunas 'Nome' e 'cor preferida'.

        Retorna:
        --------
        pandas.DataFrame
            DataFrame contendo os dados pré-processados.
        """
        self.df.reset_index(drop=True, inplace=True)
        self.df.drop(columns=['IP Address'], inplace=True)
        self.df['Added Time'] = pd.to_datetime(self.df['Added Time']).dt.strftime('%d/%m/%y - %H:%M:%S')
        self.df['Nome Codificado'] = self.le_nome.fit_transform(self.df['Nome'])
        self.df['Cor Preferida Codificada'] = self.le_cor.fit_transform(self.df['cor preferida'])
        return self.df

    def get_encoders(self):
        """
        Retorna os codificadores de 'Nome' e 'cor preferida'.

        Retorna:
        --------
        tuple
            Codificadores de 'Nome' e 'cor preferida'.
        """
        return self.le_nome, self.le_cor


class ModelTrainer:
    def __init__(self, df, le_nome, le_cor):
        """
        Inicializa a classe ModelTrainer com o DataFrame e os label encoders.

        :param df: DataFrame contendo os dados.
        :param le_nome: LabelEncoder para os nomes.
        :param le_cor: LabelEncoder para as cores.
        """
        self.df = df
        self.le_nome = le_nome
        self.le_cor = le_cor
        self.best_model = None
        self.accuracy = None
        self.cor_prevista = None

    def split_data(self):
        """
        Divide os dados em conjuntos de treino e teste.

        :return: Conjuntos de treino e teste para as variáveis independentes e dependentes.
        """
        X = self.df[['idade', 'Nome Codificado']]
        y = self.df['Cor Preferida Codificada']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        """
        Treina o modelo RandomForestClassifier usando GridSearchCV para encontrar os melhores hiperparâmetros.

        :param X_train: Conjunto de treino para as variáveis independentes.
        :param y_train: Conjunto de treino para a variável dependente.
        :raises ValueError: Se os dados de treino contiverem valores nulos.
        """
        if X_train.isnull().sum().sum() > 0 or y_train.isnull().sum() > 0:
            raise ValueError("Os dados de treino contêm valores nulos.")
        
        tamanho_dados = len(self.df)
        if tamanho_dados < 1000:
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        else:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 20, 30],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4]
            }

        grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_

    def evaluate_model(self, X_test, y_test):
        """
        Avalia o modelo treinado no conjunto de teste e calcula a acurácia.

        :param X_test: Conjunto de teste para as variáveis independentes.
        :param y_test: Conjunto de teste para a variável dependente.
        """
        y_pred = self.best_model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        print(f'Acurácia do modelo: {self.accuracy * 100:.2f}%')

    def most_chosen_color(self):
        """
        Identifica a cor mais escolhida e a idade com mais ocorrências dessa cor.
        """
        cor_mais_escolhida_codificada = self.df['Cor Preferida Codificada'].mode()[0]
        cor_mais_escolhida = self.le_cor.inverse_transform([cor_mais_escolhida_codificada])[0]
        df_filtrado = self.df[self.df['Cor Preferida Codificada'] == cor_mais_escolhida_codificada]
        ocorrencias_por_idade = df_filtrado['idade'].value_counts()
        idade_mais_comum = ocorrencias_por_idade.idxmax()
        ocorrencias = ocorrencias_por_idade.max()
        print(f'A cor mais escolhida é: {cor_mais_escolhida}')
        print(f'A idade com mais ocorrências da cor mais escolhida é: {idade_mais_comum} com {ocorrencias} ocorrências')

    def predict_new_data(self, idade, nome):
        """
        Faz a previsão da cor preferida para novos dados.

        :param idade: Idade do novo dado.
        :param nome: Nome do novo dado.
        :return: Acurácia do modelo e cor prevista.
        """
        novo_dado = pd.DataFrame({'idade': [idade], 'Nome Codificado': [self.le_nome.transform([nome])[0]]})
        cor_prevista_codificada = self.best_model.predict(novo_dado)
        self.cor_prevista = self.le_cor.inverse_transform(cor_prevista_codificada)
        print(f'Cor prevista: {self.cor_prevista[0]}')
        return self.accuracy, self.cor_prevista[0]
