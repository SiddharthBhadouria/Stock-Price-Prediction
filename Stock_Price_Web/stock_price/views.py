from django.shortcuts import render





def home(request):
    return render(request, 'stock_price/home.html')

def stocks(request):
    return render(request, 'stock_price/stocks.html')

def about(request):
    return render(request, 'stock_price/about.html')

def predictValues(request):
    stock_ticker_symbol = request.GET['stock_ticket']
    pred_val = futureValues(stock_ticker_symbol)

    return render(request, 'stock_price/predictions.html', {'price': pred_val} )


import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense , LSTM
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')

def futureValues(stock_ticker):
    
    end_date_time = datetime.now() - timedelta(1)
    end_date = end_date_time.date()

    data = yf.download(stock_ticker, start = '2020-01-01', end = end_date)
    plt.figure(figsize=(8,8))
    plt.title("Close Price History")
    plt.plot(data['Close'])
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Close Price', fontsize = 18)
    # plt.show()
    # create a new dataframe with only close column
    new_data = data.filter(['Close'])
    # convert the dataframe into a numpy array
    dataset = new_data.values
    # get the number of rows to train model on
    training_data_len = math.ceil(len(dataset)* .8)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:training_data_len, :]
    # split the data into x_train and y_train datasets
    x_train = []    # independent variables
    y_train = []    # dependent variables

    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i <= 62:
            # print(x_train)
            # print(y_train)
            print()

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1],1)))
    model.add(LSTM(50, return_sequences = False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer = "adam", loss= 'mean_squared_error')
    model.fit(x_train, y_train, batch_size = 1, epochs = 1)
    test_data = scaled_data[training_data_len-60:, :]
    # create thedatasets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:,:]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60: i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions) # unscaling the values
    rmse = np.sqrt(np.mean( predictions- y_test)**2)
    # plot the data
    train = data[:training_data_len]
    valid = data[training_data_len: ]
    valid['Predictions'] = predictions
    # visualize the model
    plt.figure(figsize = (8,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
    # plt.show()
    # get the quote
    apple_quote = yf.download(stock_ticker, start = '2020-01-01', end = end_date + timedelta(1))
    new_df = apple_quote.filter(['Close'])
    # get the last 60 day closing data values and the conver the dataframe to an array
    last_60_days = new_df[60:].values
    # scale the data to be values b/w 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)
    # creat an empty list 
    X_test = []
    # append pas 60 dasy to X_test list
    X_test.append(last_60_days_scaled)
    # convert the X_test dataset to numpy array 
    X_test = np.array(X_test)
    # reshape the data 
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # get the predicted scaled price
    pred_price = model.predict(X_test)
    # undo the scaling
    pred_price = scaler.inverse_transform(pred_price)
    return pred_price


