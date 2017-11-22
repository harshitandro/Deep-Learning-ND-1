import pickle

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import keras as ks

x = open("/mnt/Fast_Workspace/Deep-Learning-1/dataset/dogs-vs-cats/img.pkl",'rb')
y = open("/mnt/Fast_Workspace/Deep-Learning-1/dataset/dogs-vs-cats/label.pkl",'rb')

# Image data loaded from pickle file
img , labels = pickle.load(x),pickle.load(y)
print("Total Images in dataset : {}".format(img.size//img[0].size))
x.close()
y.close()

# Total Classes
n_classes = len(set(labels))
print(f"Total Classes Found : {n_classes}")

# Model for CNN

model = ks.models.Sequential()
model.add(ks.layers.Convolution2D(32,3,3,input_shape=img[0].shape))
model.add(ks.layers.Activation('relu'))
model.add(ks.layers.Convolution2D(32,1,1))
model.add(ks.layers.Activation('relu'))
model.add(ks.layers.MaxPooling2D(pool_size=(2,2)))

model.add(ks.layers.Convolution2D(32,3,3))
model.add(ks.layers.Activation('relu'))
model.add(ks.layers.MaxPooling2D(pool_size=(2,2)))

model.add(ks.layers.Convolution2D(64,3,3))
model.add(ks.layers.Activation('relu'))
model.add(ks.layers.MaxPooling2D(pool_size=(2,2)))

model.add(ks.layers.Flatten())
model.add(ks.layers.Dense(64))
model.add(ks.layers.Activation('relu'))
model.add(ks.layers.Dropout(0.5))
model.add(ks.layers.Dense(1))
model.add(ks.layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Binary encoding the labels
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)

# Hyper Parameters
nb_epoch = 30
nb_train_samples = 2048
nb_validation_samples = 832


# Train , Validation & Test split
img_test = img[-1000:]
labels_test = labels[-1000:]
img = img[:-1000]
labels = labels[:-1000]
img_train, img_val, labels_train , labels_val = train_test_split(img,labels, test_size=0.20)

# Generators for train & validation data
# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow(img_train,labels_train,batch_size=16)
val_gen = datagen.flow(img_val,labels_val)
test_gen = datagen.flow(img_test,labels_test)

model.fit_generator(
        train_gen,
        samples_per_epoch=nb_train_samples,
        epochs=nb_epoch,
        validation_data=val_gen,
        nb_val_samples=nb_validation_samples,
        verbose=2)

acc = model.evaluate_generator(test_gen, 1000)
print("Test Set Accuracy : {}".format(acc[1]))

# Save Model Parameters
model.save('models/basic_cnn_{:4}'.format(acc[1]))
