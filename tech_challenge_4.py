import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import requests
import datetime

# Função para formatar datas no formato brasileiro
def format_date(date):
    return date.strftime('%d/%m/%Y')

# Adicionar seletores de intervalo temporal para a visualização dos dados
st.sidebar.subheader('Selecionar Intervalo Temporal para Visualização')
start_date = st.sidebar.date_input('Data Inicial', value=pd.to_datetime('2023-01-01'), format='DD/MM/YYYY')
end_date = st.sidebar.date_input('Data Final', value=pd.to_datetime('2024-12-31'), format='DD/MM/YYYY')

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

# Filtro para selecionar dados dentro do intervalo escolhido pelo usuário
filtered_df = df[start_date:end_date]

# Displaying data and description within the selected dates
st.write("Visualização dos Dados:", filtered_df.head().rename(columns={'period': 'Período', 'duoarea': 'Área', 'area-name': 'Nome da Área', 'product': 'Produto', 'product-name': 'Nome do Produto', 'process': 'Processo', 'process-name': 'Nome do Processo'}))
st.write("Descrição Estatística dos Dados:", filtered_df[['Preço']].describe().rename(index={'count': 'Contagem', 'mean': 'Média', 'std': 'Desvio Padrão', 'min': 'Mínimo', '25%': '25%', '50%': 'Mediana', '75%': '75%', 'max': 'Máximo'}, columns={'Preço': 'Preço'}))

# Plotting the data within the selected dates
st.subheader("Análise Temporal dos Preços do Petróleo Brent")
fig, ax = plt.subplots()
ax.plot(filtered_df.index, filtered_df['Preço'], marker='o', linestyle='-', color='b')
ax.set_title('Tendência dos Preços do Petróleo Brent')
ax.set_xlabel('Data')
ax.set_ylabel('Preço (USD por barril)')
plt.xticks(rotation=45)
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
    result = seasonal_decompose(filtered_df['Preço'], model='additive', period=period)
    fig2, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
    result.observed.plot(ax=ax1, legend=False)
    ax1.set_ylabel('Preço')
    ax1.set_title('Preço')
    result.trend.plot(ax=ax2, legend=False)
    ax2.set_ylabel('Tendência')
    ax2.set_title('Tendência')
    result.seasonal.plot(ax=ax3, legend=False)
    ax3.set_ylabel('Sazonalidade')
    ax3.set_title('Sazonalidade')
    result.resid.plot(ax=ax4, legend=False)
    ax4.set_ylabel('Resíduo')
    ax4.set_title('Resíduo')
    plt.xticks(rotation=45)
    st.pyplot(fig2)
except ValueError as e:
    st.error(f"Erro ao decompor a série temporal: {e}")

# Dickey-Fuller test for selected dates
result_df = adfuller(filtered_df['Preço'])
st.write('Estatística ADF: {}'.format(result_df[0]))
st.write('p-valor: {}'.format(result_df[1]))
st.write('Valores Críticos:')
for key, value in result_df[4].items():
    st.write('\t{}: {:.3f}'.format(key, value))

# Autocorrelation and partial autocorrelation for selected dates
st.subheader("Autocorrelação e Autocorrelação Parcial")
fig3, ax = plt.subplots()
plot_acf(filtered_df['Preço'], ax=ax)
st.pyplot(fig3)

fig4, ax = plt.subplots()
plot_pacf(filtered_df['Preço'], ax=ax)
st.pyplot(fig4)

# Configuração e ajuste do modelo ARIMA com validação cruzada
train_df = df[df.index >= '2013-01-01']
train_df = train_df[train_df.index < '2024-01-01']

if not train_df.empty:
    tscv = TimeSeriesSplit(n_splits=5)
    best_aic = float("inf")
    best_order = None
    best_mdl = None
    
    # Procurando pelo melhor conjunto de parâmetros (p,d,q) usando validação cruzada
    for p in range(5):
        for d in range(2):
            for q in range(5):
                for train_index, test_index in tscv.split(train_df):
                    train_fold, test_fold = train_df.iloc[train_index], train_df.iloc[test_index]
                    try:
                        tmp_mdl = ARIMA(train_fold['Preço'], order=(p, d, q)).fit()
                        tmp_aic = tmp_mdl.aic
                        if tmp_aic < best_aic:
                            best_aic = tmp_aic
                            best_order = (p, d, q)
                            best_mdl = tmp_mdl
                    except:
                        continue
    
    fitted_model = best_mdl
    st.write(f"Melhor Modelo ARIMA{best_order} - AIC:{best_aic}")
    st.write(fitted_model.summary().tables[1].as_html(), unsafe_allow_html=True)  # Melhorando a exibição do sumário

    # Forecasting for 2024
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    future = pd.DataFrame(index=dates, columns=df.columns)
    future['forecast'] = fitted_model.predict(start='2024-01-01', end='2024-12-31', dynamic=True)

    # Joining training data with forecasts
    full_df = pd.concat([train_df, future])

    # Convertendo datas para o formato brasileiro
    future.index = pd.to_datetime(future.index)

    # Plotting the forecasts
    st.subheader("Previsões do Modelo ARIMA para 2024")
    fig5, ax = plt.subplots()
    ax.plot(train_df.index, train_df['Preço'], label='Preço Real (a partir de 2013)')
    ax.plot(future.index, future['forecast'], label='Previsão para 2024', color='red')
    ax.set_title('Previsões do Modelo ARIMA para o Petróleo Brent em 2024')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço (USD por barril)')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig5)
else:
    st.error('Erro: O intervalo de datas selecionado não contém dados suficientes para treinar o modelo ARIMA.')

# Exibir gráfico apenas com previsões de 2024
st.subheader("Previsão para 2024")
fig6, ax = plt.subplots()
ax.plot(future.index, future['forecast'], label='Previsão para 2024', color='red')
ax.set_title('Previsão do Modelo ARIMA para o Petróleo Brent em 2024')
ax.set_xlabel('Data')
ax.set_ylabel('Preço (USD por barril)')
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig6)
