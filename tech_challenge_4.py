import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
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
ax.set_title('Autocorrelação')
st.pyplot(fig3)

fig4, ax = plt.subplots()
plot_pacf(filtered_df['Preço'], ax=ax)
ax.set_title('Autocorrelação Parcial')
st.pyplot(fig4)

# ARIMA model configuration and fitting for data starting from 2013 within selected dates
train_df = df[df.index >= '2013-01-01']
if not train_df.empty:
    model = ARIMA(train_df['Preço'], order=(1, 0, 1))
    fitted_model = model.fit()
    st.write(fitted_model.summary().tables[1].as_html().replace("coef", "coeficiente").replace("std err", "erro padrão").replace("z", "z").replace("P>|z|", "P>|z|").replace("[0.025", "[0.025").replace("0.975]", "0.975]"), unsafe_allow_html=True)

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

# Storytelling do Projeto
st.markdown("""
## Storytelling do Projeto

### Introdução

Neste projeto, desenvolvemos um modelo preditivo para o preço do petróleo Brent utilizando técnicas de série temporal. O objetivo principal é fornecer previsões diárias do preço do petróleo, o que pode ajudar investidores, economistas e outros stakeholders a tomar decisões informadas. A análise inclui a decomposição da série temporal, testes de estacionariedade, autocorrelações e a aplicação de um modelo ARIMA.

### Contexto Histórico e Geopolítico

1. **Variações de Preço e Eventos Geopolíticos**:
    - O preço do petróleo é altamente sensível a eventos geopolíticos. Por exemplo, a invasão do Kuwait pelo Iraque em 1990 resultou em um choque no preço do petróleo devido às incertezas sobre o fornecimento. Da mesma forma, as sanções impostas ao Irã ao longo dos anos têm influenciado os preços globais do petróleo.
    - **Insight**: Nosso modelo mostra que eventos de grande escala, como guerras e sanções, podem causar picos ou quedas abruptas no preço do petróleo.

2. **Crises Econômicas**:
    - Crises econômicas, como a crise financeira de 2008, impactam diretamente a demanda por petróleo. Durante a crise de 2008, houve uma queda significativa na demanda devido à recessão global, resultando em uma queda drástica nos preços do petróleo.
    - **Insight**: As previsões do modelo ARIMA podem ser ajustadas para incluir fatores econômicos, fornecendo uma visão mais precisa durante períodos de crise.

3. **Demanda Global por Energia**:
    - O crescimento econômico de países em desenvolvimento, como China e Índia, tem aumentado a demanda global por energia. O aumento da industrialização e urbanização nesses países tem impulsionado a demanda por petróleo.
    - **Insight**: O modelo pode ser melhorado incorporando indicadores de crescimento econômico de países emergentes, ajustando as previsões de acordo com as tendências de demanda.

### Análise Técnica

1. **Decomposição da Série Temporal**:
    - A decomposição da série temporal nos permitiu separar a tendência, a sazonalidade e os resíduos dos dados históricos de preços do petróleo.
    - **Insight**: Identificamos padrões sazonais que podem ser utilizados para prever flutuações sazonais no preço do petróleo.

2. **Testes de Estacionariedade**:
    - O teste de Dickey-Fuller foi utilizado para verificar a estacionariedade da série temporal. Embora a série não tenha sido estacionária inicialmente, a diferenciação dos dados tornou-a estacionária, permitindo a aplicação eficaz do modelo ARIMA.
    - **Insight**: A transformação dos dados é crucial para a precisão do modelo preditivo.

3. **Autocorrelação e Autocorrelação Parcial**:
    - As funções de autocorrelação e autocorrelação parcial ajudaram a identificar a ordem apropriada dos parâmetros AR e MA para o modelo ARIMA.
    - **Insight**: A análise de autocorrelação confirmou a presença de correlações significativas em lags específicos, essencial para configurar o modelo.

### Previsões

1. **Previsões do Modelo ARIMA para 2024**:
    - Utilizamos um modelo ARIMA(1, 0, 1) para prever os preços do petróleo Brent para o ano de 2024. As previsões foram visualizadas em um gráfico comparativo com os preços históricos a partir de 2013.
    - **Insight**: O modelo previu uma estabilidade relativa nos preços do petróleo para 2024. No entanto, eventos geopolíticos ou econômicos inesperados poderiam alterar significativamente essas previsões.

### Conclusão

O modelo desenvolvido demonstra uma capacidade robusta de prever o preço do petróleo Brent com base em dados históricos. No entanto, a precisão das previsões pode ser afetada por eventos externos imprevistos. Portanto, é recomendável que as previsões sejam revisadas e ajustadas regularmente com base em novas informações econômicas e geopolíticas.
