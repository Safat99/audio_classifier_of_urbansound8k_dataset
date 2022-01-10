# with this code first I have imported the features for CNN and then we have imported our model. 
# next I have imported a demo file from internet to detect. Then with the prediction function the short audio has been detected. 
# Later I will make this project for real time detection. 

import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
import pandas as pd

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

	input_file = 'gunshot.wav'
	result = prediction(input_file)
	print(result)
	
	
	
	
