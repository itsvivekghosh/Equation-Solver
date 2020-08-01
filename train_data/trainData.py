import keras
import tensorflow as tf
print(tf.__version__)
import numpy as np
import pandas as pd
import pickle



def readFile(dataset_dir):
	file_name = 'train_final.csv'
	df = pd.read_csv('{}/{}'.format(dataset_dir, file_name), index_col = False)
	return df

def prepareModel(RANDOM_SEED, NUM_CLASSES):

	np.random.seed(RANDOM_SEED)
	model = keras.models.Sequential()
	model.add(keras.layers.Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu', data_format='channels_first'))
	model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.Conv2D(15, (2, 2), activation='relu'))
	model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.Dropout(0.33))

	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(128, activation='relu'))
	model.add(keras.layers.Dense(64, activation='relu'))
	model.add(keras.layers.Dense(32, activation='relu'))
	model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
	model.compile(
		loss=keras.losses.categorical_crossentropy, 
		optimizer = keras.optimizers.Adam(lr = 0.001),
		metrics=['accuracy']
	)
	return model



def saveModel(model, model_savedir="../models", savefile_model_name = 'final_model'):

	model.save_weights('{}/{}.h5'.format(model_savedir, savefile_model_name))
	model_json = model.to_json()

	with open("{}/{}.json".format(model_savedir, savefile_model_name), "w") as file:
		file.write(model_json)

	print("Model Saved Successfully as {}!".format(savefile_model_name))



def trainImages(dataset_dir = '.', model_savedir="../models", savefile_model_name = 'final_model'):

	NUM_CLASSES = 13
	RANDOM_SEED = 100
	BATCH_SIZE = 1200
	VERBOSES = 3
	EPOCHS = 10

	df = readFile(dataset_dir)

	labels = df[['784']]
	labels = np.array(labels)

	df.drop(df.columns[[784]], axis=1, inplace=True)

	from keras.utils.np_utils import to_categorical
	cat = to_categorical(labels, num_classes=NUM_CLASSES)

	dataset = []
	for i in range(len(df)):
		dataset.append(np.array(df[i : i+1]).reshape(1, 28, 28))
	dataset = np.array(dataset)

	
	model = prepareModel(RANDOM_SEED, NUM_CLASSES)
	print(model.summary())

	# model.fit(model = model, dataset, labels, epochs=EPOCHS, batch_size = BATCH_SIZE, verbose = VERBOSES)

	saveModel(model, model_savedir, savefile_model_name)


def main():
	trainImages()


if __name__ == '__main__':
	main()