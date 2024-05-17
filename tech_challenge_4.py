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

# Filtro para selecionar dados dentro do intervalo escolhido pelo usuário
filtered_df = df[start_date:end_date]

# Displaying data and description within the selected dates
st.write("Visualização dos Dados:", filtered_df.head())
st.write("Descrição Estatística dos Dados:", filtered_df.describe())

# Plotting the data within the selected dates
st.subheader("Análise Temporal dos Preços do Petróleo Brent")
fig, ax = plt.subplots()
ax.plot(filtered_df.index, filtered_df['Price'], marker='o', linestyle='-', color='b')
ax.set_title('Tendência dos Preços do Petróleo Brent')
ax.set_xlabel('Data')
ax.set_ylabel('Preço (USD por barril)')
st.pyplot(fig)

# Ajustar dinamicamente o período para decomposição sazonal
if len(filtered_df) >= 730:
    period = 365
elif len(filtered_df) >= 60:
    period = 30
else:
    period = 7

# Seasonal decomposition for selected dates
st.subheader("Decomposição da Série Temporal")
try:
    result = seasonal_decompose(filtered_df['Price'], model='additive', period=period)
    fig2 = result.plot()
    st.pyplot(fig2)
except ValueError as e:
    st.error(f"Erro ao decompor a série temporal: {e}")

# Dickey-Fuller test for selected dates
result_df = adfuller(filtered_df['Price'])
st.write('ADF Statistic: {}'.format(result_df[0]))
st.write('p-value: {}'.format(result_df[1]))
st.write('Critical Values:')
for key, value in result_df[4].items():
    st.write('\t{}: {:.3f}'.format(key, value))

# Autocorrelation and partial autocorrelation for selected dates
st.subheader("Autocorrelação e Autocorrelação Parcial")
fig3, ax = plt.subplots()
plot_acf(filtered_df['Price'], ax=ax)
st.pyplot(fig3)

fig4, ax = plt.subplots()
plot_pacf(filtered_df['Price'], ax=ax)
st.pyplot(fig4)

# ARIMA model configuration and fitting for data before 2024 within selected dates
train_df = filtered_df[filtered_df.index < '2024-01-01']
model = ARIMA(train_df['Price'], order=(1, 0, 1))
fitted_model = model.fit()
st.write(fitted_model.summary())

# Forecasting for 2024
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
future = pd.DataFrame(index=dates, columns=df.columns)
future['forecast'] = fitted_model.predict(start='2024-01-01', end='2024-12-31', dynamic=True)

# Joining training data with forecasts
full_df = pd.concat([train_df, future])

# Plotting the forecasts
st.subheader("Previsões do Modelo ARIMA para 2024")
fig5, ax = plt.subplots()
ax.plot(train_df.index, train_df['Price'], label='Preço Real (até 2023)')
ax.plot(future.index, future['forecast'], label='Previsão para 2024', color='red')
ax.set_title('Previsões do Modelo ARIMA para o Petróleo Brent em 2024')
ax.set_xlabel('Data')
ax.set_ylabel('Preço (USD por barril)')
ax.legend()
st.pyplot(fig5)
