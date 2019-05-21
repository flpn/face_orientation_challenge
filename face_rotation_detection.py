import os
import shutil
import math

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Convolution2D, MaxPooling2D


def build_directory_structure(folder_name, data):    
    for index, row in data.iterrows():
        directory = 'datasets/{}/{}'.format(folder_name,row['label'])
        file_directory = 'original_data/train/{}'.format(row['fn'])
        new_file_directory = '{}/{}'.format(directory, row['fn'])
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        shutil.copy(file_directory, new_file_directory)


def build_cnn_model():
    classifier = Sequential()
    classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    classifier.add(MaxPooling2D())
    classifier.add(Convolution2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D())
    classifier.add(Convolution2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D())
    classifier.add(Flatten())
    classifier.add(Dense(64, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(4, activation='softmax'))
    
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return classifier


def generate_csv(classes, prediction):
	predicted_class_indices = np.argmax(prediction, axis=1)
	labels = (classes)
	labels = dict((v,k) for k,v in labels.items())
	predictions = [labels[k] for k in predicted_class_indices]
	file_names = list(map(lambda x: x.replace('test/', ''), test_set.filenames)) 
	
	results = pd.DataFrame({'fn': file_names, 'labels': predictions})
	results.to_csv("results.csv",index=False)


if __name__ == '__main__':
	df = pd.read_csv('train.truth.csv')

	training_size = math.ceil(len(df)* 0.8)
	training_set = df[:training_size]
	validation_set = df[training_size:]

	build_directory_structure('training_set', training_set)
	build_directory_structure('valid_set', validation_set)

	train_datagen = ImageDataGenerator(rescale=1./255)
	valid_datagen = ImageDataGenerator(rescale=1./255)
	test_datagen = ImageDataGenerator(rescale=1./255)

	training_set = train_datagen.flow_from_directory(
	    'datasets/training_set',
	    target_size=(64, 64),
	    batch_size=32,
        seed=42,
	    class_mode='categorical')

	valid_set = valid_datagen.flow_from_directory(
	    'datasets/valid_set/',
	    target_size=(64, 64),
	    batch_size=32,
        seed=42,
	    class_mode='categorical')

	test_set = test_datagen.flow_from_directory(
        'original_data/',
        classes=['test'],
        target_size=(64, 64),
        seed=42,
        class_mode=None,
        batch_size=1)

	test_set.reset()

	classifier = build_cnn_model()
	
	classifier.fit_generator(
	    training_set,
	    epochs=10,
	    steps_per_epoch=1222,
	    validation_data=valid_set,
	    validation_steps=305)

	prediction = classifier.predict_generator(test_set, 5361)
	generate_csv(training_set.class_indices, prediction)
