import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import requests
import datetime

# Adicionar seletores de intervalo temporal para a visualização dos dados
st.sidebar.subheader('Selecionar Intervalo Temporal para Visualização')
start_date = st.sidebar.date_input('Data Inicial', value=pd.to_datetime('2023-01-01'), format="DD/MM/YYYY")
end_date = st.sidebar.date_input('Data Final', value=pd.to_datetime('2024-12-31'), format="DD/MM/YYYY")

@st.cache_data
def load_data():
    api_key = 'llflpOIMWYDhjqfbWUj8bg1bCpdlccFikD1zBJoQ'
    base_url = 'https://api.eia.gov/v2/petroleum/pri/spt/data/'
    params = {
        'api_key': api_key,
        'frequency': 'daily',
        'data[0]': 'value',
        'facets[product][]': 'EPCBRENT',
        'offset': 0,
        'length': 5000
    }
    all_data = []
    while True:
        response = requests.get(base_url, params=params)
        data = response.json()
        all_data.extend(data['response']['data'])
        if params['offset'] + 5000 >= int(data['response']['total']):
            break
        params['offset'] += 5000
    df = pd.DataFrame(all_data)
    df['Data'] = pd.to_datetime(df['period'])
    df['Preço'] = df['value'].astype(float)
    df.set_index('Data', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df = df.asfreq('D')
    df['Preço'] = df['Preço'].interpolate()  # Tratando valores ausentes
    return df

df = load_data()

# Exibindo dados e descrição
st.write("Visualização dos Dados:", df[['Preço']].head())
st.write("Descrição Estatística dos Dados:", df[['Preço']].describe().transpose())

# Filtrando os dados com base nas datas selecionadas
filtered_df = df.loc[start_date:end_date]

# Análise Temporal
st.subheader("Análise Temporal dos Preços do Petróleo Brent")
fig, ax = plt.subplots()
ax.plot(filtered_df.index, filtered_df['Preço'], marker='o', linestyle='-', color='b')
ax.set_title('Tendência dos Preços do Petróleo Brent')
ax.set_xlabel('Data')
ax.set_ylabel('Preço (USD por barril)')
st.pyplot(fig)

# Decomposição da Série Temporal
st.subheader("Decomposição da Série Temporal")
result = seasonal_decompose(filtered_df['Preço'], model='additive', period=365)
fig2 = result.plot()
fig2.suptitle('Decomposição da Série Temporal')
fig2.axes[0].set_ylabel('Preço')
fig2.axes[1].set_ylabel('Tendência')
fig2.axes[2].set_ylabel('Sazonal')
fig2.axes[3].set_ylabel('Resíduo')
st.pyplot(fig2)

# Teste de Dickey-Fuller
st.subheader("Teste de Dickey-Fuller")
result_df = adfuller(filtered_df['Preço'])
st.write('Estatística ADF: {}'.format(result_df[0]))
st.write('p-valor: {}'.format(result_df[1]))
st.write('Valores Críticos:')
for key, value in result_df[4].items():
    st.write(f'\t{key}: {value:.3f}')

# Autocorrelação e Autocorrelação Parcial
st.subheader("Autocorrelação e Autocorrelação Parcial")
fig3, ax = plt.subplots()
plot_acf(filtered_df['Preço'], ax=ax)
ax.set_title('Autocorrelação')
st.pyplot(fig3)

fig4, ax = plt.subplots()
plot_pacf(filtered_df['Preço'], ax=ax)
ax.set_title('Autocorrelação Parcial')
st.pyplot(fig4)

# Configuração e ajuste do modelo ARIMA
train_df = filtered_df[filtered_df.index < '2024-01-01']
model = ARIMA(train_df['Preço'], order=(1, 0, 1))
fitted_model = model.fit()
st.subheader("Resumo do Modelo ARIMA")
st.write(fitted_model.summary())

# Previsões para 2024
start_date_pred = '2024-01-01'
end_date_pred = '2024-12-31'
dates = pd.date_range(start=start_date_pred, end=end_date_pred, freq='D')
future = pd.DataFrame(index=dates, columns=filtered_df.columns)
future['forecast'] = fitted_model.predict(start=start_date_pred, end=end_date_pred, dynamic=True)

# Juntando os dados de treino com as previsões
full_df = pd.concat([train_df, future])

# Plotando previsões do ARIMA
st.subheader("Previsões do Modelo ARIMA para 2024")
fig5, ax = plt.subplots()
ax.plot(train_df.index, train_df['Preço'], label='Preço Real (até 2023)')
ax.plot(future.index, future['forecast'], label='Previsão para 2024', color='red')
ax.set_title('Previsões do Modelo ARIMA para o Petróleo Brent em 2024')
ax.set_xlabel('Data')
ax.set_ylabel('Preço (USD por barril)')
ax.legend()
st.pyplot(fig5)

# Gráfico apenas com as previsões para 2024
st.subheader("Previsão do Modelo ARIMA para 2024")
fig6, ax = plt.subplots()
ax.plot(future.index, future['forecast'], color='red', label='Previsão para 2024')
ax.set_title('Previsão do Modelo ARIMA para o Petróleo Brent em 2024')
ax.set_xlabel('Data')
ax.set_ylabel('Preço (USD por barril)')
ax.legend()
st.pyplot(fig6)
