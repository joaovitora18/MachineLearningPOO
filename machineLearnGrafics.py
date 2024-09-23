import matplotlib.pyplot as plt

class Graficos:
    """
    Classe para criar e salvar gráficos de ocorrências por idade e cor, e resultados de modelo.
    """

    def __init__(self, df, le_cor):
        self.df = df
        self.le_cor = le_cor

    def grafico_ocorrencias_por_idade(self, ocorrencias_por_idade):
        """
        Cria e salva o gráfico de ocorrências por idade.
        """
        plt.figure(figsize=(10, 6))
        ocorrencias_por_idade.sort_index().plot(kind='bar', color='skyblue')
        plt.title('Ocorrências por Idade')
        plt.xlabel('Idade')
        plt.ylabel('Número de Ocorrências')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', linewidth=0.7)
        plt.tight_layout()
        plt.savefig('./img/ocorrencias_por_idade.png')
        plt.close()
        print("Gráfico de ocorrências por idade criado com sucesso!")

    def grafico_ocorrencias_por_cor(self):
        """
        Cria e salva o gráfico de ocorrências por cor.
        """
        ocorrencias_por_cor = self.df['Cor Preferida Codificada'].value_counts()
        ocorrencias_por_cor.index = self.le_cor.inverse_transform(ocorrencias_por_cor.index)
        plt.figure(figsize=(10, 6))
        ocorrencias_por_cor.plot(kind='bar', color='lightcoral')
        plt.title('Ocorrências por Cor Preferida')
        plt.xlabel('Cor')
        plt.ylabel('Número de Ocorrências')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', linewidth=0.7)
        plt.tight_layout()
        plt.savefig('./img/ocorrencias_por_cor.png')
        plt.close()
        print("Gráfico de ocorrências por cor criado com sucesso!")

    def resultado_modelo(self, accuracy, cor_prevista):
        """
        Cria e salva a figura com o resultado do modelo.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        texto = f'Acurácia do modelo: {accuracy * 100:.2f}%\nCor prevista para o exemplo: {cor_prevista}'
        ax.text(0.5, 0.5, texto, ha='center', va='center', fontsize=20)
        plt.savefig('./img/resultado_modelo.png', bbox_inches='tight')
        plt.close(fig)
