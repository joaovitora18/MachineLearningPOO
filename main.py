from machineLearnClasses import DataLoader, DataPreprocessor, ModelTrainer
from machineLearnTables import Tabela
from machineLearnGrafics import Graficos

# Exemplo de uso
data_loader = DataLoader("result.xlsx")
df = data_loader.get_data()

data_preprocessor = DataPreprocessor(df)
df_preprocessed = data_preprocessor.preprocess()
le_nome, le_cor = data_preprocessor.get_encoders()

model_trainer = ModelTrainer(df_preprocessed, le_nome, le_cor)
X_train, X_test, y_train, y_test = model_trainer.split_data()
model_trainer.train_model(X_train, y_train)
model_trainer.evaluate_model(X_test, y_test)
model_trainer.most_chosen_color()
accuracy, cor_prevista = model_trainer.predict_new_data(20, 'Joao, Vitor')

# Trabalhando com tabelas
tabela = Tabela(df)
tabela.estilizar_tabela()

df_ordenado = df.sort_values(by='Nome')  # Exemplo de ordenação
tabela.ordenar_tabela(df_ordenado)

df_filtrado = df[df['idade'] > 21]  # Exemplo de filtragem
tabela.filtrar_tabela(df_filtrado)

df_final = df.drop(columns=['Nome Codificado', 'Cor Preferida Codificada'])
tabela.tabela_completa(df_final)

# Trabalhando com gráficos
graficos = Graficos(df, le_cor)

ocorrencias_por_idade = df['idade'].value_counts()
graficos.grafico_ocorrencias_por_idade(ocorrencias_por_idade)

graficos.grafico_ocorrencias_por_cor()

# Usando a acurácia e a cor prevista do modelo
graficos.resultado_modelo(accuracy, cor_prevista)
