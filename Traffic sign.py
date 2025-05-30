import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameters
path = "myData"
labelFile = "labels.csv"
batch_size_val = 50
steps_per_epoch_val = 2000 #This means each epoch processes 2000 * 50 = 100,000 images (augmented)
epochs_val= 10
imageDimensions = (32,32,3) # Initial dimensions before grayscale
testRatio = 0.2
validationRatio = 0.2


# importing of images
count=0
images=[]
classNo=[]
myList=os.listdir(path)
print("Total Classes Detected: ",len(myList))
noOfClasses = len(myList)
print("Importing Classes....")
for x in range(0,len(myList)):
    myPicList=os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        images.append(curImg)
        classNo.append(count)
    print(count,end=" ")
    count+=1
print(" ")
images= np.array(images)
classNo = np.array(classNo)


#Split Data
X_train,X_test,y_train,y_test = train_test_split(images,classNo,test_size=testRatio)
X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train, test_size = validationRatio)

# X_train = array of images to train
# y_train = corresponding class ID

#To check if number of images matches to number of labels for each data set
print("Data Shapes")
print("Train",end= "")
print(X_train.shape,y_train.shape)
print("Validation",end="")
print(X_validation.shape,y_validation.shape)
print("Test",end="")
print(X_test.shape,y_test.shape)
assert(X_train.shape[0]==y_train.shape[0]),"The number of images is not equal to the number of labels in training set"
assert(X_validation.shape[0]==y_validation.shape[0]),"The number of images is not equal to the number of labels in validation set"
assert(X_test.shape[0]==y_test.shape[0]), "The number of images is not equal to the number of labels in test set"
assert(X_train.shape[1:]==(imageDimensions))," The dimensions of the Training images are wrong "
assert(X_validation.shape[1:]==(imageDimensions))," The dimensions of the Validation images are wrong "
assert(X_test.shape[1:]==(imageDimensions))," The dimensions of the Test images are wrong"

#read csv file
data=pd.read_csv(labelFile)
print("data shape ",data.shape,type(data))

#Display some sample images of all the classes
num_of_samples=[]
cols=5
num_classes =noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(cols*2, num_classes*2))
fig.tight_layout()
for i in range(cols):
    for j,row in data.iterrows():
        if j < num_classes and np.any(y_train == j): # Check if class j is present
            x_selected = X_train[y_train == j]
            if len(x_selected) > 0:
                 axs[j][i].imshow(cv2.cvtColor(x_selected[random.randint(0, len(x_selected)- 1), :, :], cv2.COLOR_BGR2RGB)) # Use cvtColor for matplotib
                 axs[j][i].axis("off")
                 if i == 2: # Set title only for the middle column
                     axs[j][i].set_title(str(j)+ "-"+row["Name"])
            else: # If no samples for this class in y_train (after split), keep axis off
                axs[j][i].axis("off")
        else: # If class j is out of bounds for axs or not in y_train
            if j < num_classes : # Check if j is a valid row for axs
                 axs[j][i].axis("off")


# This part needs to be after the plot because it relies on iterating through `data.iterrows()` which has `num_classes` rows
# Recalculate num_of_samples based on the actual y_train distribution
num_of_samples = []
for j in range(num_classes):
    num_of_samples.append(np.sum(y_train == j))


# display bar chart showing no of samples for each category
print(num_of_samples)
plt.figure(figsize=(12,4))
plt.bar(range(0,num_classes),num_of_samples)
plt.title("Distribution of the training set")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

# preprocessing the images
def grayscale(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img=cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img=grayscale(img) #convert to grayscale
    img=equalize(img)  #standardize the lighting in an image
    img=img/255        # normalization of values b/w 0 and 1
    return img
X_train=np.array(list(map(preprocessing,X_train)))
X_validation=np.array(list(map(preprocessing,X_validation)))
X_test=np.array(list(map(preprocessing,X_test)))



# add a depth of 1
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

# Update imageDimensions for grayscale images
processedImageDimensions = (imageDimensions[0], imageDimensions[1], 1)


dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20) # batch_size for viewing samples
X_batch, y_batch = next(batches)

# TO SHOW AUGMENTED IMAGE SAMPLES
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(processedImageDimensions[0], processedImageDimensions[1]), cmap='gray')
    axs[i].axis('off')
plt.show()

y_train = utils.to_categorical(y_train, noOfClasses)
y_validation = utils.to_categorical(y_validation, noOfClasses)
y_test = utils.to_categorical(y_test, noOfClasses)


# cnn model
def myModel():
    no_of_filters = 60
    size_of_filter = (5, 5)
    size_of_filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_of_nodes = 500
    model = Sequential()
    model.add(Conv2D(no_of_filters, size_of_filter, input_shape=processedImageDimensions,
                      activation='relu'))
    model.add(Conv2D(no_of_filters, size_of_filter, activation='relu'))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Conv2D(no_of_filters // 2, size_of_filter2, activation='relu'))
    model.add(Conv2D(no_of_filters // 2, size_of_filter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(no_of_nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))  # output layer

    # compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=optimizer)
    return model


model = myModel()
print(model.summary())



history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                    steps_per_epoch=steps_per_epoch_val,
                    epochs=epochs_val,
                    validation_data=(X_validation, y_validation),
                    shuffle=True)

# plot
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# STORE THE MODEL AS A PICKLE OBJECT
pickle_out= open("model_trained.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()
print("Model saved to model_trained.p")

