

import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
import pandas as pd

#GUI portion
import tkinter as tk
from tkinter import filedialog
from recorder import Recorder


#convert the data and labels for understandable numerical data
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split



def prepare_dataset(test_size, validation_size): 	
	#load data
	X = np.array(df.feature.tolist())
	y = np.array(df.class_label.tolist())
	
	le = LabelEncoder()
	y = to_categorical(le.fit_transform(y))
	
	#create train/test split
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = test_size)
	
	#create train/validation split
	X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = validation_size)

	#for CNN tensor flow expects a 3d array -->(130,13,1)
	X_train = X_train[...,np.newaxis] #4d array --> (num_samples, 130, 13,1)
	X_validation = X_validation[...,np.newaxis]
	X_test = X_test[...,np.newaxis]
	
	return X_train, X_validation, X_test, y_train, y_validation, y_test , le



#test the model again
def testing():
	train_error, train_accuracy = model.evaluate(X_train, y_train, verbose = 1)
	test_error, test_accuracy = model.evaluate(X_test, y_test, verbose =1)

	print("Train error: {} , Train Accuracy: {} ".format(train_error, train_accuracy))
	print("Test error: {} , Test Accuracy: {} ".format(test_error, test_accuracy)) 


#extract feature from the given audio
def extract_feature(file):
	max_pad_len = 174
	try:	
		audio, sr = librosa.load(file, res_type = 'kaiser_fast')
		mfccs = librosa.feature.mfcc(audio, sr = sr, n_mfcc =40)
		pad_width = max_pad_len - mfccs.shape[1]
		mfccs = np.pad(mfccs, pad_width =((0,0), (0,pad_width)), mode = 'constant')
	
	except Exception as e:
		print("Error happened while parsing the file", file)
		return None
		
	return mfccs

#prediction _function
def prediction(input_file):
	prediction_feature = extract_feature(input_file)

	prediction_feature = prediction_feature[np.newaxis, ... , np.newaxis]
	predicted_vector = model.predict_classes(prediction_feature)
	predicted_class = le.inverse_transform(predicted_vector)
	print("The predicted class is:", predicted_class[0], '\n') 

	return predicted_class[0]
	






if __name__ == '__main__':

	
	df = pd.read_hdf('features_from_UrbanSound_for_cnn.h5', 'df')
	
	X_train, X_validation, X_test, y_train, y_validation, y_test,le = prepare_dataset(0.25, 0.2)
	
	#load model
	model = keras.models.load_model('cnn_model_after_training.h5')
	
	########################################### GUI portion ###########
		
	r = Recorder()
	load_switch = False
	record_switch = False
			
	def load_sound():
		global sound,sound_data,load_switch
		for sound_display in frame.winfo_children():
			sound_display.destroy()
		
		sound_data = filedialog.askopenfilename(initialdir="/", title="upload a sound", filetypes=(("all files", "*.*"),("mp3 files", "*.mp3")))
		print(sound_data)
		file_name = sound_data.split('/')
		panel = tk.Label(frame, text = str(file_name[-1]).upper()).pack()
		
		#len(file_name)-1]
		load_switch = True
		print('load_switch',load_switch)
	
	
	def play_loaded_sound():
		try:
			if load_switch==True:
				r.play(sound_data)
			else:
				print("load_switch",load_switch)
		except NameError:
			print("load hoy nai!!")



	def record_audio():
			global record_switch
			r.record(4, output='temp.wav')
			record_switch = True
			 
	def play_recorded_sound():
			if record_switch == True:
				r.play('temp.wav')
				#record_switch = False
			else:
				pass

	def classify_sound():
		global load_switch, record_switch
		if load_switch == True:
			input_file = sound_data
			record_switch = False
			
		if record_switch == True:
			input_file = 'temp.wav'
			load_switch = False
			
		result = prediction(input_file)
		print(result)
		msg = tk.Label(frame, text=result).pack()


	gui = tk.Tk()
	gui.configure(background="light blue")
	gui.title("Sound Classifier GUI")
	gui.geometry("640x640")
	gui.iconbitmap("@clipping_sound.xbm")
	gui.resizable(0, 0)

	#tk.Label(gui, text = 'It\'s resizable').pack(side = tk.TOP, pady = 10)
	title = tk.Label(gui, text="UrbanSound8K Classifier", padx=25, pady=6, font=("", 12)).pack()

	canvas = tk.Canvas(gui, height=500, width=500, bg='grey', bd=3)
	canvas.pack(side = tk.TOP)

	button_rec = tk.Button(canvas, text='Record Audio for 4 seconds', fg='white', bg= 'green', command = record_audio)
	button_rec.pack(side=tk.TOP)

	play_record_button = tk.Button(gui, text = 'Play Recorded Sound', fg = 'white', bg = 'black', command = play_recorded_sound).pack(side=tk.BOTTOM)


	frame = tk.Frame(gui, bg='white')
	frame.place(relwidth=.8, relheight=0.8, relx=.1, rely=0.1)

	b = tk.Button(gui,text='Choose Sound', fg='white', bg = 'black',command=load_sound)
	b.pack(side=tk.LEFT)

	classify_sound_button = tk.Button(gui,text="Classify Sound", fg="white", bg="grey", command=classify_sound)
	classify_sound_button.pack(side=tk.RIGHT)

	play_sound_button = tk.Button(gui,text="Play Loaded Sound", fg="white", bg="grey", command = play_loaded_sound)
	play_sound_button.pack(side=tk.BOTTOM)


	gui.mainloop()
		
	#############################################
	'''if switch == True:
		print(99)
		input_file = sound_data
	#input_file = 'temp.wav'
		result = prediction(input_file)
		print(result)'''
	

	
	
	
