import re
import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier


class ChurnMLService:
    def __init__(self):
        self.script_loc = os.path.dirname(os.path.realpath(__file__))
        pass

    def preprocess(self, df):
        # Drop unnecessary columns
        df = df.drop(columns=['mrg_', 'зона_1', 'зона_2', 'регион'])

        # Feature lists
        numeric_features = [
            'сумма', 'частота_пополнения', 'доход', 'сегмент_arpu', 'частота',
            'объем_данных', 'on_net', 'продукт_1', 'продукт_2', 'секретный_скор', 'pack_freq'
        ]
        categorical_features = ['использование', 'pack']

        # Fill NaN values
        df[numeric_features] = df[numeric_features].fillna(-1)

        # Convert categorical features to lowercase
        df[categorical_features] = df[categorical_features].apply(lambda x: x.str.lower())

        # Cleaning functions
        def clean_pack(x):
            if pd.isna(x):
                return x
            x = re.sub(r"[_\-;,:=.`,~\[\]\(\)]", ' ', x)
            x = x.replace('условие', ' ').replace('test', '')
            return ' '.join([word for word in x.split() if len(word) > 1])

        def clean_usage(x):
            if pd.isna(x):
                return x
            return x.replace('ly', '').replace('>', '').replace('_', ' ')

        # Features cleaning
        df['использование'] = df['использование'].apply(clean_usage)
        df['pack'] = df['pack'].apply(clean_pack)

        # Rename pack func
        def pack_rename(x):
            if pd.isna(x):
                return 'unknown'
            keywords = {
                'секрет': 'секрет', 'промо': 'промо', 'корп': 'корп', 'выгода': 'выгода',
                'приставка': 'приставка', 'youth': 'youth', 'пилот': 'пилот', 'сон': 'сон',
                'роуминг': 'роуминг', 'безлим': 'безлим', 'временный': 'временный', 'игра': 'игра',
                'соц': 'соц', 'family': 'family', 'идея': 'идея', 'каждый': 'каждый',
                'старый': 'старый', 'input': 'input', 'cvm': 'cvm', 'сутки': 'сутки',
                'evc': 'evc', 'wifi': 'wifi'
            }
            for k, v in keywords.items():
                if k in x:
                    return v
            if 'трафик' in x:
                prices = ['490', '1000', '3000', '5000', '300', '200', '100']
                data = ['2gb', '100mb', '10gb', '1gb', '5gb', '30gb', '20gb']
                days = ['7d', '2d', '1d', 'месяц', 'неделя']
                for i in prices:
                    if str(i) not in x:
                        continue
                    for j in data:
                        if str(j) not in x:
                            continue
                        for k in days:
                            if str(k) not in x:
                                continue
                            return f"{i}_{j}_{k}"
                return 'трафик'
            if 'output' in x:
                return 'output'
            return 'unknown'

        # Rename pack
        df['pack'] = df['pack'].apply(pack_rename)

        # Файлы с энкодерами
        mean_by_packs = pd.read_csv(os.path.join(self.script_loc, 'data/mean_by_pack.csv'))
        encoded_usage = pd.read_csv(os.path.join(self.script_loc, 'data/encoded_usage.csv'))

        pack_mapping = dict(zip(mean_by_packs['pack_original'], mean_by_packs['pack']))
        usage_mapping = dict(zip(encoded_usage['использование_original'], encoded_usage['использование']))
        pack_means = dict(zip(mean_by_packs['pack'], mean_by_packs['mean_сумма_pack']))

        # Label/Mean encoders from local files
        df['pack'] = df['pack'].map(pack_mapping)
        df['использование'] = df['использование'].map(usage_mapping)
        df['mean_сумма_pack'] = df['pack'].map(pack_means)

        # Feature engineering
        df['product_sum'] = df['продукт_1'] + df['продукт_2']
        df['frequencies_ratio'] = df['частота_пополнения'] / (df['частота'] + 0.1)
        df['frequencies_sum'] = df['частота_пополнения'] + df['частота']
        df['sum_freq_ratio'] = df['сумма'] / (df['частота'] + 0.1)
        df['revenue_sum_ratio'] = df['доход'] / (df['сумма'] + 0.1)
        df['onnet_volume_ratio'] = df['on_net'] / (df['объем_данных'] + 0.1)
        df['onnet_volume_sum'] = df['on_net'] + df['объем_данных']
        df['segment_freqsum_ratio'] = df['сегмент_arpu'] / (df['frequencies_sum'] + 0.1)
        df['volume_freq_ratio'] = df['объем_данных'] / (df['частота'] + 0.1)
        df['volume_productsum_ratio'] = df['объем_данных'] / (df['product_sum'] + 0.1)
        df['onnet_productsum_ratio'] = df['on_net'] / (df['product_sum'] + 0.1)
        df['packfreq_freqsum_ratio'] = df['pack_freq'] / (df['frequencies_sum'] + 0.1)

        return df
        

    def predict(self, df):
        # Drop client id column from the dataset
        client_ids = df['client_id']
        df = df.drop(columns=['client_id'])

        # Load the catboost model
        cat = CatBoostClassifier()
        cat.load_model(self.script_loc + '/data/catboost_model_w_fe.cbm')

        # Predict probabilities of churn
        preds = cat.predict_proba(df)[:, 1]

        return client_ids, preds