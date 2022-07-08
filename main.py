import pickle
import numpy as np
print("Choose:  \n 1- Wheat prediction\n 2- barley prediction \n 3- maize predicton \n 4- sorghmn predicton")
user_choose = input("Enter: ")

if (user_choose == '1'):
    wheat_model = pickle.load(open('./models/wheat_model.sav', 'rb'))
    max_temp = int(input("Enter max temp: "))
    min_temp = int(input("Enter min temp: "))
    precipitation = int(input("Enter precipitation: "))
    humidity = int(input("Enter humidity: "))
    wind_speed = int(input("Enter wind_speed : "))
    print(
        f"Your predicton based on the data you entered is: {wheat_model.predict(np.array([max_temp, min_temp, precipitation, humidity, wind_speed]).reshape(1, 5))}")

elif (user_choose == '2'):
    barley_model = pickle.load(open('./models/barley_model.sav', 'rb'))
    max_temp = int(input("Enter max temp: "))
    min_temp = int(input("Enter min temp: "))
    precipitation = int(input("Enter precipitation: "))
    humidity = int(input("Enter humidity: "))
    wind_speed = int(input("Enter wind_speed : "))
    print(
        f"Your predicton based on the data you entered is: {barley_model.predict(np.array([max_temp, min_temp, precipitation, humidity, wind_speed]).reshape(1, 5))}")


elif (user_choose == '3'):
    maize_model = pickle.load(open('./models/maize_model.sav', 'rb'))
    max_temp = int(input("Enter max temp: "))
    min_temp = int(input("Enter min temp: "))
    precipitation = int(input("Enter precipitation: "))
    humidity = int(input("Enter humidity: "))
    wind_speed = int(input("Enter wind_speed : "))
    print(
        f"Your predicton based on the data you entered is: {maize_model.predict(np.array([max_temp, min_temp, precipitation, humidity, wind_speed]).reshape(1, 5))}")


elif (user_choose == '4'):
    sorghmn_model = pickle.load(open('./models/sorghmn_model.sav', 'rb'))
    max_temp = int(input("Enter max temp: "))
    min_temp = int(input("Enter min temp: "))
    precipitation = int(input("Enter precipitation: "))
    humidity = int(input("Enter humidity: "))
    wind_speed = int(input("Enter wind_speed : "))
    print(
        f"Your predicton based on the data you entered is: {sorghmn_model.predict(np.array([max_temp, min_temp, precipitation, humidity, wind_speed]).reshape(1, 5))}")


else:
    print("Invalid choose , Please choose from above options")
