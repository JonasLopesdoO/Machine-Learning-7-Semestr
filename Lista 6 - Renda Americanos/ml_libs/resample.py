import numpy as np
import pandas as pd

class Utils:
    def stratify(df):
        # The data will be devided in equal length
        qtde_por_classe = np.min(df['LABEL'].value_counts())
        
        # obtendo as classes da base de dados
        classes = df['LABEL'].unique()
        
        # nesta lista armazenaremos, para cada classe, um pandas.DataFrame com suas amostras
        amostras_por_classe = []
        
        for c in classes:
            # obtendo os indices do DataFrame cujas instâncias pertencem a classe c
            indices_c = df['LABEL'] == c

            # extraindo do DataFrame original as observacoes da classe c (obs_c sera um DataFrame tambem)
            obs_c = df[indices_c]

            # extraindo a amostra da classe c, caso deseje-se realizar amostragem com reposição
            # ou caso len(obs_c) < qtde_por_classe, pode-se informar o parametro replace=True
            amostra_c = obs_c.sample(qtde_por_classe)

            # armazenando a amostra_c na lista de amostras
            amostras_por_classe.append(amostra_c)

        # concatenando as amostras de cada classe em um único DataFrame
        df_final = pd.concat(amostras_por_classe)
        return df_final
    
    def split_stratified_train_test(df, perc_train, seed):
        X = df.values[:, :14]
        y = df.values[:, 14]
        
        rs = np.random.RandomState(seed)
        shuffled_indices = rs.permutation(X.shape[0])
        
        ## Updating X and y to new shuffled values
        X = X[shuffled_indices]
        y = y[shuffled_indices]
        
        n = int(round(len(y) * perc_train))
        X_train = X[:n]
        y_train = y[:n]
        X_test = X[n:]
        y_test = y[n:]
        
        dados = []
        dados.append(X_train)
        dados.append(y_train)
        dados.append(X_test)
        dados.append(y_test)
        
        return dados
    
    
    
    
    
    
    