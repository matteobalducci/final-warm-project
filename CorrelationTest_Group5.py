import pandas as pd
import openpyxl
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Carica il file Excel
df2 = pd.read_excel('GW_02.xlsx')

df_filtrato = df2[(df2['Years'] >= 1980) & (df2['Years'] <= 2019)]

# Seleziona le colonne dei dati dalla versione filtrata del DataFrame
variabile_dipendente = df_filtrato['Annual Mean']
variabile_indipendente = df_filtrato['MtCO2e']

# Esegui il test di correlazione di Pearson
correlazione, p_value = pearsonr(variabile_dipendente, variabile_indipendente)

print(f'Coefficienti di Correlazione di Pearson: {correlazione}')
print(f'P-value: {p_value}')

variabile_indipendente_1 = df_filtrato['Total population by sex']

# Esegui il test di correlazione di Pearson
correlazione_1, p_value_1 = pearsonr(variabile_dipendente, variabile_indipendente_1)

print(f'Coefficienti di Correlazione di Pearson: {correlazione_1}')
print(f'P-value: {p_value_1}')

variabile_indipendente_2 = df_filtrato['GDP']

# Esegui il test di correlazione di Pearson
correlazione_2, p_value_2 = pearsonr(variabile_dipendente, variabile_indipendente_2)

print(f'Coefficienti di Correlazione di Pearson: {correlazione_2}')
print(f'P-value: {p_value_2}')