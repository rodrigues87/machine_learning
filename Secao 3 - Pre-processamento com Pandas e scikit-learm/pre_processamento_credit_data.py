##################################### PADRONIZAÇÃO DE DADOS ##########################################
"""
Created on Mon Nov 13 12:29:01 2017

@author: Jones
"""
import pandas as pd
import numpy as np
base = pd.read_csv('credit_data.csv')
#mostra estatisticas sobre os dados 
#no terminal, o comando base informa o conteudo do arquivo
base.describe()


base.loc[base['age'] < 0]
"""
# apagar a coluna
base.drop('age', 1, inplace=True)

"""
# apagar somente os registros com problema
#base.drop(base[base.age < 0].index, inplace=True)
# preencher os valores manualmente
# preencher os valores com a média
base.mean()
base['age'].mean()
base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = 40.92
        
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

#1 AO 4 SIGNIFICA QUE IRÁ ATÉ O 3
##IDENTIFICANDO AS COLUNAS DOS ATRIBUTOS PREVISORES E DA CLASSE
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

##ADICIONANDO MÉDIAS AOS LOCAIS QUE NÃO POSSUEM VALOR "NAN"
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:,0:3])


##ESCALONANDO OS VALORES PARA FACILITAR O PROCESSO DE APRENDIZAGEM
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

                  
                  
####################################################################################################
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  