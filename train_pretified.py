# -*- coding: utf-8 -*-
"""train_pretified.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-nyhpTUowxC2MTgZXR1CeaNpZbb1R1eb
"""

import numpy as np
import pandas as pd
import random,os
import sys
from sklearn.preprocessing import StandardScaler, RobustScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from IPython.display import display

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything()
pd.set_option('display.max_columns', None)

#загрузка датасета
train=pd.read_csv('covid_data_train.csv')
test=pd.read_csv('covid_data_test.csv')
test['Unnamed: 1']=test['Unnamed: 0']
test.set_index('Unnamed: 1',inplace=True)

nans = train[train.population.isna()]
train = train.drop(nans.index,axis=0)

columns_to_drop_70 = [] #список столбцов, в которых содержание NaN >= 70%
for c in train.columns:
    if train[c].isna().sum()/len(train)>=0.7:
        columns_to_drop_70.append(c)


categoricals = ['district', 'subject']

ivls_ekmos = ['ivl_per_100k', 'ivl_number', 'ekmo_per_100k', 'ekmo_number']

tuberculs = ['num_patients_tubercul_1992','num_patients_tubercul_1993','num_patients_tubercul_1994','num_patients_tubercul_1995',
           'num_patients_tubercul_1996','num_patients_tubercul_1997','num_patients_tubercul_1998','num_patients_tubercul_1999',
           'num_patients_tubercul_2000','num_patients_tubercul_2001','num_patients_tubercul_2002','num_patients_tubercul_2003',
           'num_patients_tubercul_2004','num_patients_tubercul_2005','num_patients_tubercul_2006','num_patients_tubercul_2007',
           'num_patients_tubercul_2008','num_patients_tubercul_2009','num_patients_tubercul_2010','num_patients_tubercul_2011',
           'num_patients_tubercul_2012','num_patients_tubercul_2013','num_patients_tubercul_2014','num_patients_tubercul_2015',
           'num_patients_tubercul_2016','num_patients_tubercul_2017']

transports = ['epirank_bus','epirank_train','epirank_bus_cat','epirank_train_cat','epirank_bus',
            'epirank_train','epirank_bus_cat','epirank_train_cat']

subjects = ['Хакасия', 'Оренбургская область', 'Краснодарский край', 'Татарстан', 'Ростовская область', 'Свердловская область', 
          'Чувашия', 'Якутия', 'Алтайский край', 'Владимирская область', 'Пермский край', 'Белгородская область', 'Тульская область', 
          'Иркутская область', 'Крым', 'Чукотский АО', 'Тверская область', 'Кемеровская область', 'Мурманская область', 'Чечня', 
          'Мордовия', 'Нижегородская область', 'Саратовская область', 'Приморский край', 'Красноярский край', 'Архангельская область', 
          'Томская область', 'Астраханская область', 'Челябинская область', 'Вологодская область', 'Бурятия', 'Кабардино-Балкария', 
          'Калужская область', 'Московская область', 'Забайкальский край', 'Новосибирская область', 'Ульяновская область', 
          'Пензенская область', 'Амурская область', 'Карелия', 'Башкортостан', 'Ханты-Мансийский АО — Югра', 'Северная Осетия — Алания', 
          'Хабаровский край', 'Еврейская АО', 'Ставропольский край', 'Воронежская область', 'Ленинградская область', 'Орловская область', 
          'Новгородская область', 'Брянская область', 'Костромская область', 'Смоленская область', 'Псковская область', 'Ивановская область', 
          'Волгоградская область', 'Марий Эл', 'Коми', 'Удмуртия', 'Кировская область', 'Ярославская область', 'Алтай', 'Калмыкия', 
          'Липецкая область', 'Ямало-Ненецкий АО', 'Калининградская область', 'Курганская область', 'Дагестан', 'Сахалинская область', 
          'Курская область', 'Тамбовская область', 'Самарская область', 'Тюменская область', 'Омская область', 'Рязанская область', 
          'Тыва', 'Магаданская область', 'Адыгея', 'Москва', 'Ингушетия', 'Ненецкий АО', 'Камчатский край', 'Санкт-Петербург', 'Севастополь']

#заполнение NaN нулями, удаление ненужных столбцов
def prepare_and_clean_data(df):
    df = df.drop(['region_x','Unnamed: 0'], axis=1)
    df['has_metro'] = df['has_metro'].fillna(0)
    df = df.drop(columns_to_drop_70, axis=1)
    df[ivls_ekmos] = df[ivls_ekmos].fillna(0)
    df = df.drop(transports, axis=1)

    return df

#'умное' заполнение пустых значений для признаков, связанных с туберкулезом
def smart_fillna_for_tubercul(train, test):
    tmp = pd.concat([train, test])

    fill_vals = pd.DataFrame(tmp.groupby('name')[tuberculs])
    fill_vals_dict = {}

    for i in range(len(fill_vals)):
        fill_vals_dict[fill_vals[0][i]] = np.nanmean(fill_vals[1][i][tuberculs]) #np.nanmean(fill_vals[1][i][tuberculs]) for pandas==1.3

    for i in tmp.index:
        tmp.loc[i,tuberculs] = tmp.loc[i,tuberculs].fillna(fill_vals_dict[tmp.loc[i,'name']])
      
    tmp = tmp.drop('name',axis=1)

    for i in tmp.columns:
        if tmp[i].isna().sum()>0 and i not in categoricals:
            tmp[i] = tmp[i].fillna(np.nanmean(tmp[i]))
    
    return tmp[:len(train)], tmp[len(train):]

#создание новых признаков
def making_features(df):
    df['avg_temp_mul_hum'] = df['humidity_max'] / df['avg_temp_max']
    return df

#нормализация данных
def scaling(train, test):
    count_var = []

    tmp = pd.concat([train, test])
    for col in tmp.columns:
        if tmp[col].dtype != 'object':
            count_var.append(col)

    df_tmp = tmp[(count_var)]
    std_scaler = StandardScaler()
    std_scaler.fit(df_tmp)

    scaled_train = std_scaler.transform(train[count_var])
    scaled_test = std_scaler.transform(test[count_var])

    count_var_col = []
    for col_name in count_var:
        count_var_col.append(col_name + '_std')

    train[count_var_col] = scaled_train
    test[count_var_col]  = scaled_test

    train = train.drop(count_var, axis=1)
    test = test.drop(count_var, axis=1)

    return train, test

train = prepare_and_clean_data(train)
test = prepare_and_clean_data(test)

y = train['inf_rate']
train = train.drop('inf_rate', axis=1)
test = test.drop('inf_rate', axis=1)

train, test = smart_fillna_for_tubercul(train, test)

train = making_features(train)
test = making_features(test)

train, test = scaling(train, test)

tmp = pd.concat([train, test])
tmp = pd.get_dummies(tmp)

train, test = tmp[:len(train)], tmp[len(train):]

}X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.2, random_state=42, shuffle=False)

#модель для экспериментов
cb_for_valid = CatBoostRegressor(iterations=13000, eval_metric='MAE', random_seed=42, random_strength=0.6, learning_rate=0.007)
cb_for_valid.fit(X_train, y_train, verbose=500, eval_set=(X_val, y_val))
print(mean_absolute_error(y_val, cb_for_valid.predict(X_val)))

#важность признаков
imp = cb_for_valid.get_feature_importance(prettified=True).set_index('Feature Id')
display(imp)

#обучение на полном датасете финальной модели для прода
final_cb = CatBoostRegressor(iterations=13000, random_seed=42, random_strength=0.6, learning_rate=0.007)
final_cb.fit(train, y, verbose=500)

#предсказания 
test_preds = final_cb.predict(test)
sub = pd.DataFrame({'Unnamed: 0': test.index, 'inf_rate': test_preds})
print(sub[:5])

sub.to_csv('FSD.csv', index=False)