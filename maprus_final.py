from urllib.request import urlopen
import json
import geopandas as gpd
import altair as alt
import pandas as pd
import requests
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup as bs
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn import preprocessing
label=preprocessing.LabelEncoder()


def map(sub):
    train = pd.read_csv('C:\\Users\\stani\\PycharmProjects\\covid-19\\covid_data_train_final.csv')

    for i in range(len(train['inf_rate'])):
        if train['inf_rate'].isna()[i]:
            train = train.drop(i, axis=0)
    train = train.reset_index()


    with urlopen(
            'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/russia.geojson') as response:
        counties = json.load(response)
    gdf = gpd.GeoDataFrame(counties['features'])
    df = pd.DataFrame(columns=list(gdf['properties'][0].keys()), index=[*range(0, 83)])
    for i in range(len(gdf['properties'])):
        for j in df.columns:
            df[j][i] = gdf['properties'][i][j]
    df['coordinates'] = 0
    df['type'] = 0
    df = df.drop(['created_at', 'updated_at'], axis=1)

    for i in range(len(gdf['geometry'])):
        for j in gdf['geometry'][0].keys():
            df[j][i] = gdf['geometry'][i][j]
    df['inf_rate'] = 0.0
    for i in range(len(df['name'])):
        df['inf_rate'][i] = train[train.subject == train.subject[i]]['inf_rate'].mean()

    gdf = gpd.GeoDataFrame(df)
    fig = go.Figure(go.Choroplethmapbox(geojson=counties,
                                        locations=gdf['name'],
                                        z=sub['inf_rate'],
                                        featureidkey="properties.name",
                                        colorscale='reds',
                                        customdata=np.stack((sub['inf_rate'])).T,
                                        zmin=0,
                                        zmax=5,
                                        marker_opacity=0.5,
                                        marker_line_width=1,
                                        hovertemplate='Частота заражения: %{customdata}'))
    fig.update_layout(mapbox_style="open-street-map",
                  mapbox_zoom=2,mapbox_center = {"lat":60.18678 , "lon": 95.857324} )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig