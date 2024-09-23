import matplotlib.pyplot as plt

class Tabela:
    """
    Classe para renderizar e salvar tabelas estilizadas, ordenadas e filtradas como PNG.
    """

    def __init__(self, df):
        self.df = df

    def estilizar_tabela(self):
        """
        Estiliza a tabela com um gradiente de cor e salva como PNG.
        """
        styled_table = self.df.style.background_gradient(cmap='viridis')
        self._salvar_tabela(styled_table.data.values, styled_table.data.columns, './img/tabela_estilizada.png')

    def ordenar_tabela(self, df_ordenado):
        """
        Renderiza a tabela ordenada e salva como PNG.
        """
        self._salvar_tabela(df_ordenado.values, df_ordenado.columns, './img/tabela_ordenada.png')

    def filtrar_tabela(self, df_filtrado):
        """
        Renderiza a tabela filtrada e salva como PNG.
        """
        self._salvar_tabela(df_filtrado.values, df_filtrado.columns, './img/tabela_filtrada.png')

    def tabela_completa(self, df_final):
        """
        Renderiza a tabela completa e salva como PNG.
        """
        self._salvar_tabela(df_final.values, df_final.columns, './img/tabela_completa.png')

    def _salvar_tabela(self, cell_text, col_labels, filename):
        """
        Método auxiliar para salvar a tabela como PNG.
        """
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(2, 2)
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
