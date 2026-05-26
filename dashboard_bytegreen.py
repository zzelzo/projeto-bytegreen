import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(layout="wide")

st.title("Dashboard ByteGreen - Analytics")
st.write("Monitoramento de infra e calculo de energia")

# carregando dados com cache pro streamlit
@st.cache_data
def carrega_dados():
    df = pd.read_csv('dataset_bytegreen_limpo.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df['consumo_acumulado_watts'] = df['power_consumption'].cumsum()
    return df

df = carrega_dados()

# calculo integral
tempo_seg = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
potencia = df['power_consumption']

energia_joules = np.trapezoid(y=potencia.to_numpy(), x=tempo_seg.to_numpy())
energia_kwh = energia_joules / 3600000 

energia_poupada = energia_kwh * 0.15 
co2_eco = energia_poupada * 0.085

st.subheader("Impacto ESG")

c1, c2, c3 = st.columns(3)
c1.metric("Energia Total (Integral)", f"{energia_kwh:.2f} kWh")
c2.metric("CO2 Economizado", f"{co2_eco:.2f} kg")
c3.metric("Média de Consumo", f"{df['power_consumption'].mean():.2f} W") 

st.write("---")
st.subheader("Consumo ao longo do tempo")
# agrupa por dia pro grafico nao pesar e travar o navegador
df_grafico = df.set_index('timestamp')['consumo_acumulado_watts'].resample('D').max()
st.line_chart(df_grafico)

st.write("---")
st.subheader("Modelo Preditivo de Demanda (ML)")

features = ['cpu_usage', 'memory_usage', 'network_traffic']
target = 'power_consumption'

df_ml = df.dropna(subset=features + [target])
X = df_ml[features]
y = df_ml[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

st.write("Simular nova VM:")
col1, col2, col3 = st.columns(3)
sim_cpu = col1.slider("CPU (%)", 0.0, 100.0, 50.0)
sim_mem = col2.slider("RAM (%)", 0.0, 100.0, 50.0)
sim_net = col3.number_input("Rede (MB)", value=500)

nova_maq = pd.DataFrame({'cpu_usage': [sim_cpu], 'memory_usage': [sim_mem], 'network_traffic': [sim_net]})
prev = modelo.predict(nova_maq)[0]

# feedback visual da previsao
st.success(f"Consumo Previsto: {prev:.2f} W")