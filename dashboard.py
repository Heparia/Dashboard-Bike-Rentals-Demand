import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import textwrap

day_df = pd.read_csv('day.csv', delimiter=',')
hour_df = pd.read_csv('hour.csv', delimiter=',')

day_df['dteday'] = pd.to_datetime(day_df['dteday'])
hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])

columnOutliers = ['hum', 'windspeed', 'casual']
for col in columnOutliers:
    Q1 = day_df[col].quantile(0.25)
    Q3 = day_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    day_df[col] = day_df[col].clip(lower=lower_bound, upper=upper_bound)

columnOutliers = ['hum', 'windspeed', 'casual', 'registered', 'cnt']
for col in columnOutliers:
    Q1 = hour_df[col].quantile(0.25)
    Q3 = hour_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    hour_df[col] = hour_df[col].clip(lower=lower_bound, upper=upper_bound)

def weatherImpact(df, bycolumn):
  w_impact_df = day_df[[bycolumn, 'casual', 'registered', 'cnt']].groupby(by=bycolumn).agg({
      'casual': 'median',
      'registered': 'median',
      'cnt': 'median',
  })
  return w_impact_df

def denormalize_temperature(normalized_temp, t_min, t_max):
  return normalized_temp * (t_max - t_min) + t_min

def denormalize_humidity(normalized_humidity):
    return normalized_humidity * 100

def denormalize_wind(wind):
    return wind * 67

t_min = -8
t_max = 39

at_min=-16
at_max=50

def scatter_model_1(df, columnName, color):
    plt.figure()
    return sns.scatterplot(data=df, x=df.index, y=df[columnName], color=color, label=columnName)

def scatter_model_2(df, columnName, color, xlabel, glabel):
    sns.scatterplot(data=df, x=df.index, y=df[columnName], color=color, label=glabel)
    plt.xlabel(xlabel)
    plt.ylabel("Bike Rentals Demand")

def scatter_model_3(nama_data, xlabel):
    plt.figure()
    scatter_model_2(nama_data, 'casual', 'royalblue', xlabel, "Casual")
    scatter_model_2(nama_data, 'registered', 'orange', xlabel, "Registered")
    scatter_model_2(nama_data, 'cnt', 'forestgreen', xlabel, "Total")

def impact_scatter_chart(nama_data, xlabel):
    plt.figure()
    plt.title(xlabel, fontsize=30, pad=20)
    scatter_model_3(nama_data, xlabel)
    return plt.gcf() 

st.set_page_config(layout="wide")
st.title(
    """
    Data Analysis Project: Bike Sharing Dataset (2011-2012)
    """
)
st.divider()

cuacaPertama = "Clear, Few clouds, Partly cloudy, Partly cloudy"
cuacaKedua = "Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist"
cuacaKetiga = "Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds"
cuacaKeempat = "Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog"

weather_description = {
    1: cuacaPertama,
    2: cuacaKedua,
    3: cuacaKetiga
}

all_weather_description = {
    1: cuacaPertama,
    2: cuacaKedua,
    3: cuacaKetiga,
    4: cuacaKeempat
}

col1, col2 = st.columns([3,2])

with col1:


    new_row = pd.DataFrame({
        'Weather_Description': [all_weather_description[3]],
        'casual': [0],
        'registered': [0],
        'cnt': [0]
    })
    st.header("# Weather Impact on Bike Rental Demand")
    option = st.selectbox(
            "Method",
            ("Median", "Sum"),
        )
        
    global weathersit_impact_df
    weathersit_impact_df = day_df[['weathersit', 'casual', 'registered', 'cnt']].groupby(by='weathersit')
    if option.lower() == "median":
        weathersit_impact_df = weathersit_impact_df.median()
    else :
        weathersit_impact_df = weathersit_impact_df.sum()
    
    weathersit_impact_df['Weather_Description'] = weathersit_impact_df.index.map(weather_description)
    weathersit_impact_df = pd.concat([weathersit_impact_df, new_row], ignore_index=True)

    categories = weathersit_impact_df.index
    categories_caption = weathersit_impact_df['Weather_Description']
    casual = weathersit_impact_df['casual']
    registered = weathersit_impact_df['registered']
    both = weathersit_impact_df['cnt']

    def wrap_labels(labels, width=20):
        return [textwrap.fill(label, width=width) for label in labels]

    wrapped_labels = wrap_labels(categories_caption, width=30)

    x = range(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar([p - width for p in x], casual, width, label='Casual', color='royalblue')
    bars2 = ax.bar(x, registered, width, label='Registered', color='orange')
    bars3 = ax.bar([p + width for p in x], both, width, label='Both', color='forestgreen')

    # Menambahkan label ke setiap batang
    def add_labels(bars):
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    # Menambahkan label dan judul
    ax.set_xlabel('Weather Conditions')
    ax.set_ylabel('Bike Rentals Demand')
    ax.set_title('Casual vs Registered vs Both Users by Weather Conditions')
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_labels)
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Weather Conditions")
    st.write("1. ", all_weather_description[1])
    st.write("2. ", all_weather_description[2])
    st.write("3. ", all_weather_description[3])
    st.write("4. ", all_weather_description[4])
    col3, col4 = st.columns([1,1])
    with col3:
        max_values = weathersit_impact_df[['casual', 'registered', 'cnt']].max().astype(int)
        max_labels = weathersit_impact_df[['casual', 'registered', 'cnt']].idxmax()
        max_summary = pd.DataFrame({
            'Total': max_values,
            'Weathersit Label': max_labels+1
        })
        max_summary.index = ['Casual Users', 'Registered Users', 'Both Users']
        st.subheader("Max Value")
        st.table(max_summary)
    with col4:
        min_values = weathersit_impact_df[['casual', 'registered', 'cnt']].min().astype(int)
        min_labels = weathersit_impact_df[['casual', 'registered', 'cnt']].idxmin()
        min_summary = pd.DataFrame({
            'Total': min_values,
            'Weathersit Label':  min_labels+1
        })
        min_summary.index = ['Casual Users', 'Registered Users', 'Both Users']
        st.subheader("Min Value")
        st.table(min_summary)

st.divider()

sh_impact_df = hour_df[['season', 'hr', 'casual', 'registered', 'cnt']].groupby(by='season')
sh_impact_df.head()

season_1_df = sh_impact_df.get_group(1)
season_2_df = sh_impact_df.get_group(2)
season_3_df = sh_impact_df.get_group(3)
season_4_df = sh_impact_df.get_group(4)

season_1_df = season_1_df.drop_duplicates()
season_2_df = season_2_df.drop_duplicates()
season_3_df = season_3_df.drop_duplicates()
season_4_df = season_4_df.drop_duplicates()

st.header("# Analysis of Peak Hours in Bike Rental Demand by Season")
st.caption("Method - Median")
col7, col8, col9, col10 = st.columns(4)

def line_model_1(df, ycolumn, color):
    sns.lineplot(data=df, x='hr', y=ycolumn, label=ycolumn, color=color)

def line_model_2(df, ycolumn, color, labely):
    sns.lineplot(data=df, x="hr", y=ycolumn, label=labely, color=color)
    plt.xlabel('Hour') 
    plt.ylabel('Bike Rentals Demand')

def line_model_3(nama_data):
    line_model_2(nama_data, 'casual', 'royalblue', 'Casual')
    line_model_2(nama_data, 'registered', 'orange', 'Registered')
    line_model_2(nama_data, 'cnt', 'forestgreen', 'Both')

def season_maker_2(nama_data, data):
    plt.figure()  
    plt.title(nama_data, fontsize=30, pad=20)
    line_model_3(data)
    plt.legend() 
    return plt.gcf() 

with col7:
    st.pyplot(season_maker_2("Spring", season_1_df))
with col8:
    st.pyplot(season_maker_2("Summer", season_2_df))
with col9:
    st.pyplot(season_maker_2("Autumn",season_3_df))
with col10:
    st.pyplot(season_maker_2("Winter",season_4_df))

hour_impact_df = hour_df[['season', 'hr', 'temp', 'atemp', 'hum','windspeed', 'casual', 'registered', 'cnt']].groupby(by='hr').median()

t_hour_impact_df = weatherImpact(hour_impact_df, "temp")
at_hour_impact_df = weatherImpact(hour_impact_df, "atemp")
h_hour_impact_df = weatherImpact(hour_impact_df, "hum")
w_hour_impact_df = weatherImpact(hour_impact_df, "windspeed")

t_hour_impact_df['denormalized_temp_celcius'] = denormalize_temperature(t_hour_impact_df.index, t_min, t_max)
h_hour_impact_df['humidity_denormalized'] = h_hour_impact_df.index.map(denormalize_humidity)
at_hour_impact_df['denormalized_atemp_celcius'] = denormalize_temperature(at_hour_impact_df.index, at_min, at_max)
w_hour_impact_df['humidity_denormalized'] = w_hour_impact_df.index.map(denormalize_humidity)

st.divider()

with st.expander("More Advanced..."):
    st.subheader("Specific Impact of Various Factors on Bike Rental Demand: Casual, Registered, and Total Users")
    st.caption("Method - Median | Normalized Value")
    st.write(
        "1. Temperature (temp): Normalized temperature in Celsius. The values are derived via "
        r"$\frac{t - t_{\text{min}}}{t_{\text{max}} - t_{\text{min}}}$, where $t_{\text{min}} = -8$ and $t_{\text{max}} = +39$ (only in hourly scale)."
    )
    st.write(
        "2. Feeling Temperature (atemp): Normalized feeling temperature in Celsius. The values are derived via "
        r"$\frac{t - t_{\text{min}}}{t_{\text{max}} - t_{\text{min}}}$, where $t_{\text{min}} = -16$ and $t_{\text{max}} = +50$ (only in hourly scale)."
    )
    st.write(
        "3. Humidity (hum): Normalized humidity. The values are divided by 100 (max)."
    )
    st.write(
        "4. Windspeed (windspeed): Normalized wind speed. The values are divided by 67 (max)."
    )
    col11, col12, col13, col14 = st.columns(4)
    with col11:
        st.pyplot(impact_scatter_chart(t_hour_impact_df, "Temperature"))
    with col12:
        st.pyplot(impact_scatter_chart(at_hour_impact_df, "Feeling Temperature"))
    with col13:
        st.pyplot(impact_scatter_chart(h_hour_impact_df, "Humadity"))
    with col14:
        st.pyplot(impact_scatter_chart(w_hour_impact_df, "Windspeed"))
