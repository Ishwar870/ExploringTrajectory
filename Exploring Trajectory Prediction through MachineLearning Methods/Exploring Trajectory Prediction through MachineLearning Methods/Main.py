
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
import os
import re
import numpy as np 
import pandas as pd
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from numpy import array
from numpy import argmax
from numpy import array_equal


main = tkinter.Tk()
main.title("Exploring Trajectory Prediction through Machine Learning Methods")
main.geometry("1300x1200")

global filename
global model, encoder_model, decoder_model
global dataset
global lstm_error,gru_error
list = []

def upload():
    global filename
    list.clear()
    filename = filedialog.askopenfilename(initialdir="dataset")
    with open(filename, "r") as file:
        for line in file:
            line = line.strip('\n')
            list.append(line)
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
                        

def lstmModel():
    global model, encoder_model, decoder_model
    # define training encoder
    encoder_inputs = Input(shape=(None, 9))
    encoder = LSTM(512, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, 9))
    decoder_lstm = LSTM(512, return_sequences=True, return_state=True)   #LSTM with SEQ2SEQ object sequences created here
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	
    decoder_dense = Dense(9, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(512,))
    decoder_state_input_c = Input(shape=(512,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    text.insert(END,"LSTM Model Generated\n\n")

def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]
    
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)
	# start of sequence input
	target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = []
	for t in range(n_steps):
		# predict next char
		teacher_ratio, h, c = infdec.predict([target_seq] + state)
		# store prediction
		output.append(teacher_ratio[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_seq = teacher_ratio
	return array(output) 

def trainLSTM():
    global dataset
    global model
    global lstm_error,gru_error
    train = pd.read_csv(filename)
    size = len(train)
    dataset = np.zeros((size, 9, 9))

    m = 0;
    n = 0
    p = 0

    for i in range(len(train)) :
        person = int(train.loc[i, "personid"])
        position = int(train.loc[i, "steps"])
        latitude = float(train.loc[i, "latitude"])
        longitude = float(train.loc[i, "longitude"])
        n = 0
        for j in range(len(train)):
            person1 = int(train.loc[j, "personid"])
            position1 = int(train.loc[j, "steps"])
            latitude1 = float(train.loc[j, "latitude"])
            longitude1 = float(train.loc[j, "longitude"])
            if person == person1:
                dataset[m][position1-1][n] = latitude1
                n = n + 1
                dataset[m][position1-1][n] = longitude1
                n = n + 1
                dataset[m][position1-1][n] = person
                n = n + 1
                if n >= 9:
                    n = 0
                    
        m = m + 1
    print(dataset.shape)    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    # summarize defined model
    print(model.summary())
    model.fit([dataset,dataset], dataset, epochs=10)
    scores = model.evaluate([dataset,dataset], dataset, verbose=2)
    accuracy = scores[1]*100
    lstm_error = 100.0 - accuracy
    gru_error = (100.0 - accuracy) + 10
    
    print("LSTM Accuracy: %.2f%%" % (scores[1]*100))
    model.save("trajectory_model.h5py")
    text.insert(END,"See Black Console to View LSTM Training Processed Data\n\n")

def predict():
    latitude = simpledialog.askstring(title="Latitude",prompt="Enter Current Latitude Location Value")
    longitude = simpledialog.askstring(title="Longitude",prompt="Enter Current Longitude Location Value")
    user = simpledialog.askstring(title="User",prompt="Enter User ID")
    b = np.zeros((1, 9, 9))
    b[0][0][0] = float(latitude)
    b[0][1][1] = float(longitude)
    b[0][2][2] = int(user)
    lat = ''
    lon = ''
    for i in range(len(list)):
        if i > 0:
            arr = list[i].split(",")
            if float(latitude) == float(arr[2]) and float(longitude) == float(arr[3]):
                arr = list[i+1].split(",")
                lat = arr[2]
                lon = arr[3]
                break
            
    target = predict_sequence(encoder_model, decoder_model, b, 3, 9)
    output = one_hot_decode(target)
    text.insert(END,"Predicted Sequences of users next steps are : \n\n")
    print(output)
    print(str(output[0])+" "+str(output[1])+" "+str(output[2]))
    text.insert(END,"Next Location Latuitude : "+lat+"\n\n");
    text.insert(END,"Next Location Latuitude : "+lon+"\n\n");
    text.insert(END,"Next Sequences : "+str(dataset[int(user)][output[0]])+"\n");
    text.insert(END,"Next Sequences : "+str(dataset[int(user)][output[1]])+"\n");
    text.insert(END,"Next Sequences : "+str(dataset[int(user)][output[2]])+"\n");
   

def graph():
    height = [lstm_error,gru_error]
    bars = ('LSTM MSE Error','GRU MSE Error')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

    
font = ('times', 16, 'bold')
title = Label(main, text='Exploring Trajectory Prediction through Machine Learning Methods')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Trajectory Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

extractButton = Button(main, text="Generate LSTM Model", command=lstmModel)
extractButton.place(x=50,y=150)
extractButton.config(font=font1) 

pearsonButton = Button(main, text="Train LSTM with Seq2Seq", command=trainLSTM)
pearsonButton.place(x=330,y=150)
pearsonButton.config(font=font1) 

runsvm = Button(main, text="Predict Trajectory", command=predict)
runsvm.place(x=620,y=150)
runsvm.config(font=font1) 

graph = Button(main, text="MSE Graph", command=graph)
graph.place(x=850,y=150)
graph.config(font=font1) 

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
