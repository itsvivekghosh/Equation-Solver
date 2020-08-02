import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from keras.layers import Convolution2D, Flatten, Activation, Dropout, Dense, MaxPooling2D
from keras.models import Sequential
print("tf.__version__ is:", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)




def readFile(dataset_dir):
	file_name = 'train_final.csv'
	df = pd.read_csv('{}/{}'.format(dataset_dir, file_name), index_col = False)
	return df



def saveModel(model, model_savedir="../models", savefile_model_name = 'final_model'):

	model.save_weights('{}/{}.h5'.format(model_savedir, savefile_model_name))
	model_json = model.to_json()

	with open("{}/{}.json".format(model_savedir, savefile_model_name), "w") as file:
		file.write(model_json)

	print("Model Saved Successfully as {}!".format(savefile_model_name))



def trainImages(dataset_dir = '.', model_savedir="../models", savefile_model_name = 'final_model'):

	NUM_CLASSES = 13
	BATCH_SIZE = 1
	VERBOSES = 1
	EPOCHS = 10

	df = readFile(dataset_dir)
	labels = df[['784']]
	labels = np.array(labels)

	df.drop(df.columns[[784]], axis=1, inplace=True)

	from keras.utils.np_utils import to_categorical
	cat = to_categorical(labels, num_classes=NUM_CLASSES)

	dataset = []
	for i in range(len(df)):
		dataset.append(np.array(df[i:i+1]).reshape(1, 28, 28))
	dataset = np.array(dataset)

	model = Sequential()
	model.add(Convolution2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu', data_format='channels_first'))
	model.add(MaxPooling2D(2,2, dim_ordering='tf'))
	model.add(Convolution2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(2,2, dim_ordering='tf'))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(NUM_CLASSES, activation='softmax'))
	model.compile(
		loss="categorical_crossentropy", 
		optimizer = "adam",
		metrics=['accuracy']
	)
	print(model.summary())

	model.fit(dataset, cat, epochs=10, batch_size=500, shuffle=True, verbose=1)

	saveModel(model, model_savedir, savefile_model_name)


def main():
	trainImages()


if __name__ == '__main__':
	main()