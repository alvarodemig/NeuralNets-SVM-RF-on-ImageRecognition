from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dropout, Dense, BatchNormalization, Input
import numpy as np
import keras, os,cv2
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import breedsHelpFunctions as hf
from keras.models import load_model, Model
from keras import regularizers
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#TXT file where the list of pictures for each set are saved
traintxt = os.getcwd() + "/BreedTrain.txt"
testtxt = os.getcwd() + "/BreedTest.txt"
valtxt = os.getcwd() + "/BreedValidation.txt"

#Pictures in color = 3, in GreyScale = 1
color = 1

#Size of the picture
picsize = [100,100]

#Batch Size, epochs to run and optimizer
batch_size = 50
epochs = 10
optimizer = 'rmsprop'
#HOG dictionary where we will save the characteristics for each HOG and datasets
HOG = {'HOG1': {'param' : [(100,100), (5,5), (10,10), (5,5), 9], 'HOGdesTrain': [], 'HOGdesTest': [], 'HOGdesVal': [], 'PCAmodel' : []},
        'HOG2': {'param' : [(100,100), (10,10), (20,20), (10,10), 9], 'HOGdesTrain': [], 'HOGdesTest': [], 'HOGdesVal': [], 'PCAmodel' : []},
        'HOG3': {'param' : [(100,100), (20,20), (40,40), (20,20), 9], 'HOGdesTrain': [], 'HOGdesTest': [], 'HOGdesVal': [], 'PCAmodel' : []},
        'HOG4': {'param' : [(100,100), (50,50), (100,100), (50,50), 9], 'HOGdesTrain': [], 'HOGdesTest': [], 'HOGdesVal': [], 'PCAmodel' : []}}

#Dataset load, using the help function created and choosing if we want to include more pictures than the basics
X_train, X_test, X_val, y_train, y_test, y_val = \
    hf.datasetExpansion(traintxt, testtxt, valtxt, extratrain = [['TRANS100','TRANS100']], validsets = 'original')
print('Training sets preprocessed')

	#Creating output classes and dummies arrays to use as output
classes, y_tr = hf.create_dummies(y_train)
_, y_te = hf.create_dummies(y_test)
_, y_va = hf.create_dummies(y_val)


#Compute the HOG for each picture of each set, and save into the dictionary
for dog in X_train:
    img = cv2.imread(dog)
    for h in HOG:
        params = HOG[h]['param']
        hogcompute = hf.hogDescriptorCalculation(img, params[0], params[1], params[2], params[3], params[4])
        hoglist = []
        for e in range(hogcompute.size): hoglist += [float(hogcompute[e][0])]
        HOG[h]['HOGdesTrain'] += [hoglist]
        
for dog in X_test:
    img = cv2.imread(dog)
    for h in HOG:
        params = HOG[h]['param']
        hogcompute = hf.hogDescriptorCalculation(img, params[0], params[1], params[2], params[3], params[4])
        hoglist = []
        for e in range(hogcompute.size): hoglist += [float(hogcompute[e][0])]
        HOG[h]['HOGdesTest'] += [hoglist]

for dog in X_val:
    img = cv2.imread(dog)
    for h in HOG:
        params = HOG[h]['param']
        hogcompute = hf.hogDescriptorCalculation(img, params[0], params[1], params[2], params[3], params[4])
        hoglist = []
        for e in range(hogcompute.size): hoglist += [float(hogcompute[e][0])]
        HOG[h]['HOGdesVal'] += [hoglist]

'-----------------------------------------------------'
'----------------HOG VECTOR SIZES --------------------'
'-----------------------------------------------------'
#Check HOG vector sizes
for h in HOG:
    HOG[h]['HOGvectorLen'] = len(HOG[h]['HOGdesTrain'][0])
    print(h, len(HOG[h]['HOGdesTrain'][0]))

'-----------------------------------------------------'
'------------------------ PCA ------------------------'
'-----------------------------------------------------'

#PCA for each set of HOGs and save the new parameters. Components = 256
for h in HOG:
    HOG[h]['PCAmodel'] = hf.PCAhog(HOG[h]['HOGdesTrain'], 256)
    HOG[h]['PCAtrain'] = HOG[h]['PCAmodel'].transform(HOG[h]['HOGdesTrain'])
    HOG[h]['PCAtest'] = HOG[h]['PCAmodel'].transform(HOG[h]['HOGdesTest'])
    HOG[h]['PCAval'] = HOG[h]['PCAmodel'].transform(HOG[h]['HOGdesVal'])

print('#####################################################')
print('-------------- INPUTS & OUTPUTS ---------------------')
print('#####################################################')

#List of inputs, tests and validation sets
inputs = [HOG['HOG1']['PCAtrain'], HOG['HOG2']['PCAtrain'],HOG['HOG3']['PCAtrain'],HOG['HOG4']['HOGdesTrain']]
tests = [HOG['HOG1']['PCAtest'], HOG['HOG2']['PCAtest'],HOG['HOG3']['PCAtest'],HOG['HOG4']['HOGdesTest']]
validations = [HOG['HOG1']['PCAval'], HOG['HOG2']['PCAval'],HOG['HOG3']['PCAval'],HOG['HOG4']['HOGdesVal']]

#Number of HOG that we want to use as input into the Multilayer Neural Net
modelnumber = 1

modelinput = inputs[modelnumber-1]
testinput = tests[modelnumber-1]
valinput = validations[modelnumber-1]

#Name of the model that will be saved
name = 'HOGnoAUG120' + str(modelnumber)

print('#####################################################')
print('--------------- MODEL STRUCTURE ---------------------')
print('#####################################################')
#Creating the model structure

keras.backend.clear_session()
model = Sequential()
model.add(Dense(500, activation='relu', use_bias=True, input_shape= hf.listToMatrix(modelinput)[0].shape))
model.add(Dropout(0.3))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
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

modelHistory = model.fit(hf.listToMatrix(modelinput), y_tr,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=((hf.listToMatrix(testinput)), y_te),
    callbacks=[history, bestModelAcc, bestModelLoss])

print('#####################################################')
print('----------------- MODEL EVALUATION ------------------')
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

print('------------------------------------------------------------------------')
print('----------------------------- TEST -------------------------------------')
print('------------------------------------------------------------------------')
print('TEST: BEST ACCURACY MODEL')
_,_,_ = hf.modelEvaluation(best_model_acc, hf.listToMatrix(testinput), y_test, classes)
print('------------------------------------------------------------------------')
print('TEST: BEST LOSS MODEL')
_,_,_ = hf.modelEvaluation(best_model_loss, hf.listToMatrix(testinput), y_test, classes)
print('------------------------------------------------------------------------')
print('TEST: LAST MODEL')
_,_,_ = hf.modelEvaluation(model, hf.listToMatrix(testinput), y_test, classes)

print('------------------------------------------------------------------------')
print('-------------------------- VALIDATION ----------------------------------')
print('------------------------------------------------------------------------')
print('VALIDATION: BEST ACCURACY MODEL')
_,_,_ = hf.modelEvaluation(best_model_acc, hf.listToMatrix(valinput), y_val, classes)
print('------------------------------------------------------------------------')
print('VALIDATION: BEST LOSS MODEL')
_,_,_ = hf.modelEvaluation(best_model_loss, hf.listToMatrix(valinput), y_val, classes)
print('------------------------------------------------------------------------')
print('LAST MODEL')
_,_,_ = hf.modelEvaluation(model, hf.listToMatrix(valinput), y_val, classes)


print('------------------------------------------------------------------------')
print('------------------------ MLP ENSEMBLE ----------------------------------')
print('------------------------------------------------------------------------')
print('BEST ACCURACY MODEL')

val_prob_ens = np.zeros([len(y_val), len(classes)])
for i in range(1,5):
    bestModelPath = os.getcwd() + '/bestModel/'
    bestModelPathAcc = bestModelPath + 'model_acc_HOG' + str(i) + '.hdf5'
    bestModelPathLoss = bestModelPath + 'model_loss_HOG' + str(i) + '.hdf5'
    best_model_acc = load_model(bestModelPathAcc)
    best_model_loss = load_model(bestModelPathLoss)
    val_prob_ens += best_model_acc.predict(hf.listToMatrix(validations[i]), verbose = 0)
    val_prob_ens += best_model_loss.predict(hf.listToMatrix(validations[i]), verbose = 0)
    
pred = []
for p in val_prob_ens:
    max_value = max(p)
    max_index = np.where(p==max_value)
    pred += [classes[max_index[0][0]]]

print('MLP ENSEMBLE MODEL')
accuracy = round(accuracy_score(y_val, pred)*100,2)
print('-----------------------------\n')
print('Accuracy: ', accuracy, '%\n')
print(confusion_matrix(y_val, pred))
print(classification_report(y_val, pred))
print('-----------------------------')
