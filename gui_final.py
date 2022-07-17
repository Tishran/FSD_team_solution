import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
from catboost import CatBoostRegressor
import maprus_final
import random
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler,MinMaxScaler
label=preprocessing.LabelEncoder()


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything()
pd.set_option('display.max_columns', None)


st.set_option('deprecation.showPyplotGlobalUse', False)
model=CatBoostRegressor()
model.load_model('C:\\Users\\stani\\PycharmProjects\\covid-19\\final_model')

st.markdown(
    "<h2 style='text-align: center; color: lightgreen;'>Предсказание уровня заражения вирусом COVID-19 в городах "
    "России </h2>",
    unsafe_allow_html=True)


@st.cache(suppress_st_warning=True)
def load_data(upload):
    if upload is not None:
        dataframe = pd.read_csv(upload)

        dataframe['Unnamed: 1'] = dataframe['Unnamed: 0']
        dataframe.set_index('Unnamed: 1', inplace=True)

        nans = dataframe[dataframe.population.isna()]
        train = dataframe.drop(nans.index, axis=0)

        columns_to_drop_70 = []  # список столбцов, в которых содержание NaN >= 70%
        for c in train.columns:
            if train[c].isna().sum() / len(train) >= 0.7:
                columns_to_drop_70.append(c)

        categoricals = ['district', 'subject']

        ivls_ekmos = ['ivl_per_100k', 'ivl_number', 'ekmo_per_100k', 'ekmo_number']

        tuberculs = ['num_patients_tubercul_1992', 'num_patients_tubercul_1993', 'num_patients_tubercul_1994',
                     'num_patients_tubercul_1995',
                     'num_patients_tubercul_1996', 'num_patients_tubercul_1997', 'num_patients_tubercul_1998',
                     'num_patients_tubercul_1999',
                     'num_patients_tubercul_2000', 'num_patients_tubercul_2001', 'num_patients_tubercul_2002',
                     'num_patients_tubercul_2003',
                     'num_patients_tubercul_2004', 'num_patients_tubercul_2005', 'num_patients_tubercul_2006',
                     'num_patients_tubercul_2007',
                     'num_patients_tubercul_2008', 'num_patients_tubercul_2009', 'num_patients_tubercul_2010',
                     'num_patients_tubercul_2011',
                     'num_patients_tubercul_2012', 'num_patients_tubercul_2013', 'num_patients_tubercul_2014',
                     'num_patients_tubercul_2015',
                     'num_patients_tubercul_2016', 'num_patients_tubercul_2017']

        transports = ['epirank_bus', 'epirank_train', 'epirank_bus_cat', 'epirank_train_cat', 'epirank_bus',
                      'epirank_train', 'epirank_bus_cat', 'epirank_train_cat']

        subjects = ['Хакасия', 'Оренбургская область', 'Краснодарский край', 'Татарстан', 'Ростовская область',
                    'Свердловская область',
                    'Чувашия', 'Якутия', 'Алтайский край', 'Владимирская область', 'Пермский край',
                    'Белгородская область', 'Тульская область',
                    'Иркутская область', 'Крым', 'Чукотский АО', 'Тверская область', 'Кемеровская область',
                    'Мурманская область', 'Чечня',
                    'Мордовия', 'Нижегородская область', 'Саратовская область', 'Приморский край', 'Красноярский край',
                    'Архангельская область',
                    'Томская область', 'Астраханская область', 'Челябинская область', 'Вологодская область', 'Бурятия',
                    'Кабардино-Балкария',
                    'Калужская область', 'Московская область', 'Забайкальский край', 'Новосибирская область',
                    'Ульяновская область',
                    'Пензенская область', 'Амурская область', 'Карелия', 'Башкортостан', 'Ханты-Мансийский АО — Югра',
                    'Северная Осетия — Алания',
                    'Хабаровский край', 'Еврейская АО', 'Ставропольский край', 'Воронежская область',
                    'Ленинградская область', 'Орловская область',
                    'Новгородская область', 'Брянская область', 'Костромская область', 'Смоленская область',
                    'Псковская область', 'Ивановская область',
                    'Волгоградская область', 'Марий Эл', 'Коми', 'Удмуртия', 'Кировская область', 'Ярославская область',
                    'Алтай', 'Калмыкия',
                    'Липецкая область', 'Ямало-Ненецкий АО', 'Калининградская область', 'Курганская область',
                    'Дагестан', 'Сахалинская область',
                    'Курская область', 'Тамбовская область', 'Самарская область', 'Тюменская область', 'Омская область',
                    'Рязанская область',
                    'Тыва', 'Магаданская область', 'Адыгея', 'Москва', 'Ингушетия', 'Ненецкий АО', 'Камчатский край',
                    'Санкт-Петербург', 'Севастополь']

        def prepare_and_clean_data(df):
            df = df.drop(['region_x', 'Unnamed: 0'], axis=1)
            df['has_metro'] = df['has_metro'].fillna(0)
            df = df.drop(columns_to_drop_70, axis=1)
            df[ivls_ekmos] = df[ivls_ekmos].fillna(0)
            df = df.drop(transports, axis=1)

            return df

        def smart_fillna_for_tubercul(train):
            tmp = train.copy()

            fill_vals = pd.DataFrame(tmp.groupby('name')[tuberculs])
            fill_vals_dict = {}

            for i in range(len(fill_vals)):
                fill_vals_dict[fill_vals[0][i]] = np.nanmean(
                    fill_vals[1][i])  # np.nanmean(fill_vals[1][i][tuberculs]) for pandas==1.3

            for i in tmp.index:
                tmp.loc[i, tuberculs] = tmp.loc[i, tuberculs].fillna(fill_vals_dict[tmp.loc[i, 'name']])

            tmp = tmp.drop('name', axis=1)

            for i in tmp.columns:
                if tmp[i].isna().sum() > 0 and i not in categoricals:
                    tmp[i] = tmp[i].fillna(np.nanmean(tmp[i]))

            return tmp

        def making_features(df):
            df['avg_temp_mul_hum'] = df['humidity_max'] / df['avg_temp_max']
            return df

        def scaling(train):
            count_var = []

            tmp = train.copy()
            for col in tmp.columns:
                if tmp[col].dtype != 'object':
                    count_var.append(col)

            df_tmp = tmp[(count_var)]
            std_scaler = StandardScaler()
            std_scaler.fit(df_tmp)

            scaled_train = std_scaler.transform(train[count_var])

            count_var_col = []
            for col_name in count_var:
                count_var_col.append(col_name + '_std')

            train[count_var_col] = scaled_train


            train = train.drop(count_var, axis=1)

            return train

        def add_absent_onehot_cols(df):
            for s in subjects:
                if s not in df.columns:
                    df[f'subject_{s}'] = 0

            return df

        train = prepare_and_clean_data(train)

        datafr = train.copy()

        try:
            y = train['inf_rate']
            train = train.drop('inf_rate', axis=1)
        except:
            y = None

        train = smart_fillna_for_tubercul(train)

        train = making_features(train)

        train = scaling(train)

        train = pd.get_dummies(train)

        train = add_absent_onehot_cols(train)

        return (train, y, datafr)
    else:
        return None, None, None


uploaded_file = st.file_uploader(label='Выберите датасет:')

df, y, df_show = load_data(uploaded_file)


list_graph=[' ', 'scatter', 'bar', 'hist', 'line']
def main():


    result = st.button('Предсказать')
    if result:
        pred = model.predict(df)
        sub = pd.DataFrame({'Unnamed: 0': df.index, 'inf_rate': pred})
        map = maprus_final.map(sub)
        st.write(map)



if __name__ == "__main__":
    main()