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
start_date = st.sidebar.date_input('Data Inicial', value=pd.to_datetime('2023-01-01'))
end_date = st.sidebar.date_input('Data Final', value=pd.to_datetime('2024-12-31'))

@st.experimental_memo
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
    df['Date'] = pd.to_datetime(df['period'])
    df['Price'] = df['value'].astype(float)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df = df.asfreq('D')
    df['Price'] = df['Price'].interpolate()  # Handling missing values
    return df

df = load_data()

# Displaying data and description
st.write("Visualização dos Dados:", df.head())
st.write("Descrição Estatística dos Dados:", df.describe())

st.subheader("Análise Temporal dos Preços do Petróleo Brent")
fig, ax = plt.subplots()
ax.plot(df.index, df['Price'], marker='o', linestyle='-', color='b')
ax.set_title('Tendência dos Preços do Petróleo Brent')
ax.set_xlabel('Data')
ax.set_ylabel('Preço (USD por barril)')
st.pyplot(fig)

st.subheader("Decomposição da Série Temporal")
result = seasonal_decompose(df['Price'], model='additive', period=365)
fig2 = result.plot()
st.pyplot(fig2)

# Teste de Dickey-Fuller
result_df = adfuller(df['Price'])
st.write('ADF Statistic: {}'.format(result_df[0]))
st.write('p-value: {}'.format(result_df[1]))
st.write('Critical Values:')
for key, value in result_df[4].items():
    st.write('\t{}: {:.3f}'.format(key, value))

st.subheader("Autocorrelação e Autocorrelação Parcial")
fig3, ax = plt.subplots()
plot_acf(df['Price'], ax=ax)
st.pyplot(fig3)

fig4, ax = plt.subplots()
plot_pacf(df['Price'], ax=ax)
st.pyplot(fig4)

# Configuração e ajuste do modelo ARIMA
train_df = df[df.index < '2024-01-01']
model = ARIMA(train_df['Price'], order=(1, 0, 1))
fitted_model = model.fit()
st.write(fitted_model.summary())

# Previsões para 2024
start_date = '2024-01-01'
end_date = '2024-12-31'
dates = pd.date_range(start=start_date, end=end_date, freq='D')
future = pd.DataFrame(index=dates, columns=df.columns)
future['forecast'] = fitted_model.predict(start=start_date, end=end_date, dynamic=True)

# Juntando os dados de treino com as previsões
full_df = pd.concat([train_df, future])

st.subheader("Previsões do Modelo ARIMA para 2024")
fig5, ax = plt.subplots()
ax.plot(train_df.index, train_df['Price'], label='Preço Real (até 2023)')
ax.plot(future.index, future['forecast'], label='Previsão para 2024', color='red')
ax.set_title('Previsões do Modelo ARIMA para o Petróleo Brent em 2024')
ax.set_xlabel('Data')
ax.set_ylabel('Preço (USD por barril)')
ax.legend()
st.pyplot(fig5)

if start_date < end_date:
    filtered_df = df.loc[start_date:end_date]
    st.subheader(f"Análise Temporal dos Preços do Petróleo Brent de {start_date} até {end_date}")
    fig6, ax = plt.subplots()
    ax.plot(filtered_df.index, filtered_df['Price'], marker='o', linestyle='-', color='b')
    ax.set_title('Tendência dos Preços do Petróleo Brent')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço (USD por barril)')
    st.pyplot(fig6)
else:
    st.error('Erro: Data inicial deve ser anterior à data final.')
