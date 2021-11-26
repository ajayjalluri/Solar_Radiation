import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import base64
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

page = st.sidebar.selectbox("Select Activity", ["Introduction", "Analytics","Radiation Prediction",])
st.sidebar.text(" \n")

st.sidebar.header("Solar Energy")
img= Image.open("image1.jpg")
st.sidebar.image(img)
st.text(" \n")

st.sidebar.header("Radiation Sensor")
img= Image.open("trackso-solar-radiation-sensor.jpg")
st.sidebar.image(img)

scaler = open('solar_std.pkl', 'rb')

std = pickle.load(scaler)

knnf = open('solar_knn.pkl', 'rb')

knn = pickle.load(knnf)


df = pd.read_csv("solar.csv")



if page=="Introduction":

    st.header("Solar Radiation")
    st.text(" \n")
    img= Image.open("solar_radiation.jpg")
    st.image(img)
    st.text(" \n")

    st.subheader("Solar radiation, often called the solar resource or just sunlight, is a general term for the electromagnetic radiation emitted by the sun. Solar radiation can be captured and turned into useful forms of energy, such as heat and electricity, using a variety of technologies.")
    st.text(" \n")
    st.header("Features That Depend On Solar Radiation(watts per meter**2) Prediction : ")
    st.text(" \n")
    st.subheader("Temperature (Fahrenheit)")
    st.subheader("Humidity (percent)")
    st.subheader("Barometric pressure (Hg)")
    st.subheader("Wind direction (degrees) ")
    st.subheader("Wind speed (miles per hour) ")
    st.subheader("Sunrise/sunset(Hawaii time) ")

    st.text(" \n")
    st.text(" \n")
    st.text(" \n")
    st.text(" \n")
    st.text(" \n")
    st.text(" \n")

    st.subheader("TEAM MEMBERS")
    st.write("* Chakilam Shiva Kumar")
    st.write("* Jalluri Ajay Vamsi")
    st.write("* Amarthaluru Paavan Dileep")
    st.write("* Morla Naga Manikanta")

    st.header("Faculty Mentor : [Dr.Amarnath Bheemaraju](https://www.bmu.edu.in/faculty/dr-amarnath-bheemaraju/)")
if page == "Analytics" :

    st.header("Radiation Box Plot")
    fig = px.box(df, y="Radiation")

    st.plotly_chart(fig,use_container_width=10)

    st.header("Distribution of Radiation")
    st.text(" \n")

    fig = px.histogram(df, x="Radiation")
    st.plotly_chart(fig,use_container_width=10)


    st.header("Scatter Plot (Temperature , Humidity)")
    fig = px.scatter(df, x="Temperature", y="Humidity",
                 color='Radiation')
    st.plotly_chart(fig,use_container_width=10)
    st.header("Scatter Plot (Pressure , Humidity)")
    fig = px.scatter(df, x="Pressure", y="Humidity",
                 color='Radiation')

    st.plotly_chart(fig,use_container_width=10)
    st.header("Scatter Plot (Temperature , Pressure)")
    fig = px.scatter(df, x="Temperature", y="Pressure",
                 color='Radiation')

    st.plotly_chart(fig,use_container_width=10)

    st.header("Pearson Correlation analysis")
    img= Image.open("heatmap.jpg")
    st.image(img)

    st.header("Distribution of Pressure")

    fig = px.histogram(df, x="Pressure")
    st.plotly_chart(fig,use_container_width=10)


    st.header("Distribution of Temperature")

    fig = px.histogram(df, x="Temperature")
    st.plotly_chart(fig,use_container_width=10)

    st.header("Distribution of Humidity")
    st.text(" \n")

    fig = px.histogram(df, x="Humidity")
    st.plotly_chart(fig,use_container_width=10)



    st.header("Temperature Box Plot")
    fig = px.box(df, y="Temperature")

    st.plotly_chart(fig,use_container_width=10)

    st.header("WindDirection(Degrees) Box Plot")
    fig = px.box(df, y="WindDirection(Degrees)")

    st.plotly_chart(fig,use_container_width=10)


    st.header("Radiation Box Plot")
    fig = px.box(df, y="Radiation")

    st.plotly_chart(fig,use_container_width=10)

    st.header("Distribution of Radiation")
    st.text(" \n")

    fig = px.histogram(df, x="Radiation")
    st.plotly_chart(fig,use_container_width=10)










if page =="Radiation Prediction" :
    st.header("Solar Radiation Prediction")
    form = st.form(key='my_form2')

    x1 = form.text_input(label='Temperature')
    form.text(" \n")
    x2 = form.text_input(label='Pressure')
    form.text(" \n")
    x3 = form.text_input(label='Humidity')
    form.text(" \n")
    x4 = form.text_input(label='WindDirection(Degrees)')
    form.text(" \n")
    x5 = form.text_input(label='Speed')
    form.text(" \n")


    submit_button = form.form_submit_button(label='Predict Radiation')
    if submit_button:



        l = [[float(x1),float(x2),float(x3),float(x4),float(x5)]]

        s = std.transform(l)
        pred = knn.predict(s)[0]
        st.text(" \n")
        a = "Radiation"+ " : " +str(float(pred)) + "watts per meter^2"
        st.write(a)
