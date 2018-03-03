from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dropout, Dense, BatchNormalization
import numpy as np
import keras, os
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import breedsHelpFunctions as hf
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#Name for the Model that will be saved
name = 'CNNnoAug120'

#TXT file where the list of pictures for each set are saved
traintxt = os.getcwd() + "/BreedTrain.txt"
testtxt = os.getcwd() + "/BreedTest.txt"
valtxt = os.getcwd() + "/BreedValidation.txt"

#Pictures in color = 3, in GreyScale = 1
color = 3

#Size of the picture
picsize = [100,100]

#Batch Size, epochs to run and optimizer
batch_size = 100
epochs = 30
optimizer = 'rmsprop'
classes = 6

############################################################
#----------------------- CODE ------------------------------
############################################################


print('#####################################################')
print('--------------- LOADING DATASETS --------------------')
print('#####################################################')

''' ---------- JUST TO LOAD ------------ '''

'''
#Dataset load, using the help function created and choosing if we want to include more pictures than the basics
# extratrain: [['TRANS100','TRANS100']]
# validsets: [['GB100','GB100']]
X_train, X_test, X_val, y_train, y_test, y_val = \
    hf.datasetExpansion(traintxt, testtxt, valtxt, extratrain = '', validsets = '')
print('Training sets loaded')

#Creating the arrays that we will use as input for our CNN
X_trainPic = hf.MatrixToDataset(X_train, picsize[0], picsize[1], color)
X_testPic = hf.MatrixToDataset(X_test, picsize[0], picsize[1], color)
X_valPic = hf.MatrixToDataset(X_val, picsize[0], picsize[1], color)
print('Training sets preprocessed')

#Creating output classes and dummies arrays to use as output
classes, y_tr = hf.create_dummies(y_train)
_, y_te = hf.create_dummies(y_test)
_, y_va = hf.create_dummies(y_val)
'''

print('#################################################################')
print('------------------------ MODEL STRUCTURE ------------------------')
print('#################################################################')

#Creating the model structure
keras.backend.clear_session()

model = Sequential()

model.add(Conv2D(16, (4, 4), padding='same', use_bias=True, activation='relu', input_shape=(picsize[0], picsize[1], color)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(32, (4, 4), padding='same', use_bias=True, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(64, (4, 4), padding='same', use_bias=True, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Dropout(0.5))
#model.add(Conv2D(128, (4, 4), padding='same', use_bias=True, activation='relu'))
#model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))
model.summary()

print('Model Structure Created')


print('#####################################################')
print('--------------- MODEL CALLBACKS ---------------------')
print('#####################################################')

#Creting paths to save the model with best accuracy and loss
bestModelPath = os.getcwd() + '/bestModel/'
bestModelPathAcc = bestModelPath + 'model_acc_' + name + '.hdf5'
bestModelPathLoss = bestModelPath + 'model_loss_' + name + '.hdf5'

#Creating the checkpoint saver for our model
bestModelAcc = ModelCheckpoint(bestModelPathAcc, monitor="val_acc",
                      save_best_only=True, save_weights_only=False)
bestModelLoss = ModelCheckpoint(bestModelPathLoss, monitor="val_loss",
                      save_best_only=True, save_weights_only=False)

history=[]
history = hf.AccuracyHistory()


print('#####################################################')
print('----------------- MODEL COMPILE ---------------------')
print('#####################################################')

#optimizer depending on our previous selection to include in the model
if optimizer == 'adam':
    optCompile = keras.optimizers.Adam(lr=0.001)
else: optCompile = 'rmsprop'

#Compiling the model before running it
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer= optCompile,
              metrics=['accuracy'])
print('Model Compiled')

print('#####################################################')
print('------------------- MODEL RUN -----------------------')
print('#####################################################')

# Model Run
modelHistory = model.fit(X_trainPic, y_tr,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_testPic, y_te),
          callbacks=[history, bestModelAcc, bestModelLoss])

print('#####################################################')
print('----------------- MODEL EVALUATION ---------------------')
print('#####################################################')

#PLOT ACCURACY AND LOSS EVOLUTION FOR TRAINING AND TESTING SETS
plt.plot(range(len(history.acc)), history.acc)
plt.plot(range(len(history.acc)), history.val_acc)
plt.legend(['Train', 'Test'], loc='upper left')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

plt.plot(range(len(history.loss)), history.loss)
plt.plot(range(len(history.loss)), history.val_loss)
plt.legend(['Train', 'Test'], loc='upper left')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


#Load Accuracy and Loss models
best_model_acc = load_model(bestModelPathAcc)
best_model_loss = load_model(bestModelPathLoss)
print('Saved models loaded')

#Print the model evaluation for the Test and Validation Sets
#Include the confusion matrix and accuracy of the model
#We use the modelEvaluation help function
print('------------------------------------------------------------------------')
print('----------------------------- TEST -------------------------------------')
print('------------------------------------------------------------------------')
print('TEST: BEST ACCURACY MODEL')
_,_,_ = hf.modelEvaluation(best_model_acc, X_testPic, y_test, classes)
print('------------------------------------------------------------------------')
print('TEST: BEST LOSS MODEL')
_,_,_ = hf.modelEvaluation(best_model_loss, X_testPic, y_test, classes)
#print('------------------------------------------------------------------------')
#print('LAST MODEL')
#_,_,_ = hf.modelEvaluation(model, X_testPic, y_test, classes)

print('------------------------------------------------------------------------')
print('-------------------------- VALIDATION ----------------------------------')
print('------------------------------------------------------------------------')
print('VALIDATION: BEST ACCURACY MODEL')
_,_,_ = hf.modelEvaluation(best_model_acc, X_valPic, y_val, classes)
print('------------------------------------------------------------------------')
print('VALIDATION: BEST LOSS MODEL')
_,_,_ = hf.modelEvaluation(best_model_loss, X_valPic, y_val, classes)
#print('------------------------------------------------------------------------')
#print('LAST MODEL')
#_,_,_ = hf.modelEvaluation(model, X_valPic, y_val, classes)

print('------------------------------------------------------------------------')
print('-------------------- CNN ENSEMBLE VALIDATION ---------------------------')
print('------------------------------------------------------------------------')
#Ensemble model creation and evaluation
val_prob_ens = np.zeros([len(y_val), len(classes)])

val_prob_ens += best_model_acc.predict(X_valPic, verbose = 0)
val_prob_ens += best_model_loss.predict(X_valPic, verbose = 0)
val_prob_ens /= 2
pred = []

for p in val_prob_ens:
    max_value = max(p)
    max_index = np.where(p==max_value)
    pred += [classes[max_index[0][0]]]

print('CNN ENSEMBLE MODEL')
accuracy = round(accuracy_score(y_val, pred)*100,2)
print('-----------------------------\n')
print('Accuracy: ', accuracy, '%\n')
print(confusion_matrix(y_val, pred))
print(classification_report(y_val, pred))
print('-----------------------------')
