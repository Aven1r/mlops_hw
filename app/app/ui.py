import streamlit as st
import requests
import pandas as pd
from io import StringIO
from requests_toolbelt.multipart.encoder import MultipartEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

backend = "http://ml:8001/api/v1/ml/predict" 

def process(file_bytes, server_url: str):
    try:
        m = MultipartEncoder(fields={"file": ("filename", file_bytes)})
        r = requests.post(
            server_url, data=m, headers={"Content-Type": m.content_type},
        )
        if r.status_code == 200:
            st.write("Request successful!")
            return r.content.decode('utf-8')
        else:
            st.write(f"Request failed with status code: {r.status_code}")
            return None
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return None

# UI layout
st.title("Предсказание оттока клиентов")

# Sidebar
st.sidebar.title("Меню")
st.sidebar.image("./static/header.png", use_column_width=True)
data_file = st.sidebar.file_uploader("Загрузите данные о пользователях в формате CSV")

if data_file is not None:
    file_bytes = data_file.getvalue()
    csv_content = process(file_bytes, backend)
    
    if csv_content:
        # Load the CSV content
        df = pd.read_csv(StringIO(csv_content), index_col=0)

        tab1, tab2 = st.tabs(["Предсказания", "Графики"])

        with tab1:
            st.subheader("Предсказания")
            st.dataframe(df, use_container_width=True)

            st.subheader("Выгрузить предсказания")

            threshold = st.slider('Выберите границу вероятности с которой класс будет считаться положительным', 0.00, 1.00, 0.5, step=0.01)

            df_to_download = df.copy()
            df_to_download['pred'] = df_to_download['pred'] >= threshold
            df_to_download['pred'] = df_to_download['pred'].astype(int)

            # Button for CSV download
            csv = df_to_download.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Скачать CSV",
                data=csv,
                file_name=f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_threshold_{str(threshold).replace('.', '_')}.csv",
                mime='text/csv',
            )

        
        with tab2:
            st.subheader("График плотности предсказанных скоров")
            range_selection = st.slider('Выберите диапазон вероятностей предсказаний', 0.01, 1.0, (0.2, 0.8), 0.01)

            # Filter the DataFrame based on the selected range
            filtered_df = df[(df['pred'] >= range_selection[0]) & (df['pred'] <= range_selection[1])]

            x1 = filtered_df['pred']
            group_labels = ['Churn']

            # Create distplot with custom bin_size
            fig = ff.create_distplot([x1], group_labels, bin_size=[0.02])

            # Plot the KDE graph
            st.plotly_chart(fig, use_container_width=True)

            #
            st.subheader("Топ лучших признаков модели CatBoost")

            feature_importances = pd.read_csv('./data/feature_importances.csv')
            
            # Sort the features
            feature_importances = feature_importances.sort_values(by='importance', ascending=False)

            # Pick numbers of top features
            top_n = st.slider('Выберите количество признаков для вывода', 1, 26, 5)

            top_features = feature_importances['feature'][:top_n]
            top_importance = feature_importances['importance'][:top_n]

            # Create a bar plot for the top features
            fig = px.bar(x=top_importance, y=top_features, orientation='h', labels={'x': 'Importance', 'y': 'Feature'},
                        title='Топ признаков модели')

            # Plot the bar chart
            st.plotly_chart(fig, use_container_width=True)



