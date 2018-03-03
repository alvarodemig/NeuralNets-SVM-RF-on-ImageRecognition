'''
This code could be parametrized for SVM or RF.
The difference between the code related to SVM and RF is small due to the lack of time
'''

import os, heapq, random, cv2
import numpy as np
import breedsHelpFunctions as hf
from sklearn import svm as sksvm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

#Dictionary of the breeds and the dogs
breedDict, dogDict = hf.retrieveLabels()
breedDict.pop('breed')

#Print barplot of the quantity of appearances.
allbreeds = []
allcounts = []
for b in breedDict:
    allbreeds += [b]
    allcounts += [breedDict[b]['count']]
objects = allbreeds
y_pos = np.arange(len(objects))
performance = allcounts
print len(allcounts)
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Images')
plt.title('Number of images per breed')
plt.show()

#Breeds with the maximum and minimum number of pics
minimages = min(allcounts)
maximages = max(allcounts)
for b in breedDict:
    if breedDict[b]['count'] == minimages: print b, breedDict[b]['count']
    if breedDict[b]['count'] == maximages: print b, breedDict[b]['count']

#See the available breeds
breedList = sorted(breedDict.keys())
for b in breedList: print b

#Select the 6 breeds with more appearances
breedcount = []
for b in breedDict: breedcount += [breedDict[b]['count']]
max_counts = min(heapq.nlargest(6, breedcount))

breeds = []
for b in breedDict:
    if breedDict[b]['count'] >= max_counts: breeds += [b]

#Manual selection of breeds to classify: Example:
#breeds = ['golden_retriever', 'german_shepherd', 'malamute', 'pomeranian', 'rottweiler', 'border_collie', 'chihuahua']

#Creating a dog list with the selected breeds and shuffling order
dogList = []
for b in breeds:
    for p in breedDict[b]['pics']: dogList += [[b,p]]
random.shuffle(dogList)

dogListComplete = []
for b in allbreeds:
    for p in breedDict[b]['pics']: dogListComplete += [[b,p]]
random.shuffle(dogListComplete)

#Creating a Dataset and target
picType = 'CROPPED100' 
pathpics = os.getcwd() + '/' + picType + '/'
targetpicnamesList = []
for d in dogList: targetpicnamesList += [[d[0], pathpics + d[1]+'_CR100.jpeg']]
targetpicnamesListComplete = []
for d in dogListComplete: targetpicnamesListComplete += [[d[0], pathpics + d[1]+'_CR100.jpeg']]

picLoc = []
picLocComplete = []
picTarget = []
picTargetComplete = []

for dog in targetpicnamesList:
    picLoc += [dog[1]]
    picTarget += [dog[0]]

for dog in targetpicnamesListComplete:
    picLocComplete += [dog[1]]
    picTargetComplete += [dog[0]]

#Creating the train, test and validation sets (70%, 15%, 15%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(picLoc, picTarget, test_size=0.30)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

X_trainC, X_testC, y_trainC, y_testC = train_test_split(picLocComplete, picTargetComplete, test_size=0.30)
X_testC, X_valC, y_testC, y_valC = train_test_split(X_testC, y_testC, test_size=0.5, random_state=1)

#Saving these sets in a file to load them when necessary
#6 breeds
traindata = ''
testdata = ''
valdata = ''
for i in range(len(X_train)): traindata += X_train[i] + ' --- ' + y_train[i] + '\n'
for i in range(len(X_test)): testdata += X_test[i] + ' --- ' + y_test[i] + '\n'
for i in range(len(X_val)): valdata += X_val[i] + ' --- ' + y_val[i] + '\n'
f = open('BreedTrain.txt','w')
f.write(traindata)
f.close()
f = open('BreedTest.txt','w')
f.write(testdata)
f.close()
f = open('BreedValidation.txt','w')
f.write(valdata)
f.close()
f = open('BreedTrain.txt','w')
f.write(traindata)
f.close()
f = open('BreedTest.txt','w')
f.write(testdata)
f.close()
f = open('BreedValidation.txt','w')
f.write(valdata)
f.close()

#All the breeds
traindata = ''
testdata = ''
valdata = ''
for i in range(len(X_trainC)): traindata += X_trainC[i] + ' --- ' + y_trainC[i] + '\n'
for i in range(len(X_testC)): testdata += X_testC[i] + ' --- ' + y_testC[i] + '\n'
for i in range(len(X_valC)): valdata += X_valC[i] + ' --- ' + y_valC[i] + '\n'
f = open('BreedTrainALL.txt','w')
f.write(traindata)
f.close()
f = open('BreedTestALL.txt','w')
f.write(testdata)
f.close()
f = open('BreedValidationALL.txt','w')
f.write(valdata)
f.close()
f = open('BreedTrainALL.txt','w')
f.write(traindata)
f.close()
f = open('BreedTestALL.txt','w')
f.write(testdata)
f.close()
f = open('BreedValidationALL.txt','w')
f.write(valdata)
f.close()


'We are using a 6 breeds dataset'
#Opening the files to load the list of files to use in our sets
traintxt = os.getcwd() + "/BreedTrain.txt"
testtxt = os.getcwd() + "/BreedTest.txt"
valtxt = os.getcwd() + "/BreedValidation.txt"

#Dataset load, using the help function created and choosing if we want to include more pictures than the basics
X_train, X_test, X_val, y_train, y_test, y_val = \
    hf.datasetExpansion(traintxt, testtxt, valtxt, extratrain = [['TRANS100','TRANS100']], validsets = 'original')

#HOG dictionary where we will save the characteristics for each HOG and datasets
HOG = {'HOG1': {'param' : [(100,100), (5,5), (10,10), (5,5), 9], 'HOGdesTrain': [], 'HOGdesTest': [], 'HOGdesVal': [], 'SVM':  [],
                           'accuracy': 0, 'C': 0, 'gamma':0, 'kernel': ''},
        'HOG2': {'param' : [(100,100), (10,10), (20,20), (10,10), 9], 'HOGdesTrain': [], 'HOGdesTest': [], 'HOGdesVal': [], 'SVM':  [],
                           'accuracy': 0, 'C': 0, 'gamma':0, 'kernel': ''},
        'HOG3': {'param' : [(100,100), (20,20), (40,40), (20,20), 9], 'HOGdesTrain': [], 'HOGdesTest': [], 'HOGdesVal': [], 'SVM':  [],
                           'accuracy': 0, 'C': 0, 'gamma':0, 'kernel': ''},
        'HOG4': {'param' : [(100,100), (50,50), (100,100), (50,50), 9], 'HOGdesTrain': [], 'HOGdesTest': [], 'HOGdesVal': [], 'SVM':  [],
                           'accuracy': 0, 'C': 0, 'gamma':0, 'kernel': ''}}

'-----------------------------------------------------'
'------------------ HOG VECTORS ----------------------'
'-----------------------------------------------------'

#Compute the HOG for each picture of each set, and save into the dictionary
counter = 0
for dog in X_train:
    counter +=1
    img = cv2.imread(dog)
    for h in HOG:
        params = HOG[h]['param']
        hogcompute = hf.hogDescriptorCalculation(img, params[0], params[1], params[2], params[3], params[4])
        hoglist = []
        for e in range(hogcompute.size): hoglist += [float(hogcompute[e][0])]
        HOG[h]['HOGdesTrain'] += [hoglist]
    if counter % 100 == 0: print('train', counter)
    
counter = 0
for dog in X_test:
    counter += 1
    img = cv2.imread(dog)
    for h in HOG:
        params = HOG[h]['param']
        hogcompute = hf.hogDescriptorCalculation(img, params[0], params[1], params[2], params[3], params[4])
        hoglist = []
        for e in range(hogcompute.size): hoglist += [float(hogcompute[e][0])]
        HOG[h]['HOGdesTest'] += [hoglist]
    if counter % 100 == 0: print('test', counter)
    
counter = 0
for dog in X_val:
    counter += 1
    img = cv2.imread(dog)
    for h in HOG:
        params = HOG[h]['param']
        hogcompute = hf.hogDescriptorCalculation(img, params[0], params[1], params[2], params[3], params[4])
        hoglist = []
        for e in range(hogcompute.size): hoglist += [float(hogcompute[e][0])]
        HOG[h]['HOGdesVal'] += [hoglist]        
    if counter % 100 == 0: print('validation', counter)


'-----------------------------------------------------'
'--------------- HOG VECTOR SIZES --------------------'
'-----------------------------------------------------'
for h in HOG:
    print h, len(HOG[h]['HOGdesTrain'][0])

'-----------------------------------------------------'
'------------------------ PCA ------------------------'
'-----------------------------------------------------'
for h in ['HOG1', 'HOG2','HOG3']:
    HOG[h]['PCAmodel'] = hf.PCAhog(HOG[h]['HOGdesTrain'], 256)
    HOG[h]['PCAtrain'] = HOG[h]['PCAmodel'].transform(HOG[h]['HOGdesTrain'])
    HOG[h]['PCAtest'] = HOG[h]['PCAmodel'].transform(HOG[h]['HOGdesTest'])
    HOG[h]['PCAval'] = HOG[h]['PCAmodel'].transform(HOG[h]['HOGdesVal'])



'#####################################################'
'############# SUPPORT VECTOR MACHINE ################'
'#####################################################'

'-----------------------------------------------------'
'------------ SVM PARAMETERS -------------------------'
'-----------------------------------------------------'
C = [1, 3, 5, 7, 10]
gamma = [0.001, 0.01, 0.1, 1, 10]
kernel = ['linear','rbf', 'poly', 'sigmoid']
kernel = ['rbf','sigmoid']

'''
#SVM FOR EACH COMBINATION OF PARAMETERS AND EACH HOG.
#IT AUTOMATICALLY SELECTS THE PARAMETERS THAT OBTAIN THE HIGHEST ACCURACY IN THE TEST SET FOR EACH HOG
'''

'-----------------------------------------------------'
'------------ SVM - 6 BREEDS -------------------------'
'-----------------------------------------------------'
#SVM calculation for each HOG
for h in HOG:
        print '------------------------'
        print h
        for k in kernel:
            for c in C:
                for g in gamma:
                    clf = sksvm.SVC(C=c, gamma = g, kernel= k, decision_function_shape='ovo', probability=True, class_weight= 'balanced')
                    clf.fit(HOG[h]['HOGdesTrain'], y_train)
                    #Predict Training
                    #print 'TRAIN ', h
                    pred_prob = clf.predict_proba(HOG[h]['HOGdesTrain'])
                    pred = clf.predict(HOG[h]['HOGdesTrain'])
                    pred.tolist()
                    #print confusion_matrix(y_train, pred)
                    
                    #Predict Test
                    #print 'TEST ', h
                    pred_prob = clf.predict_proba(HOG[h]['HOGdesTest'])
                    pred = clf.predict(HOG[h]['HOGdesTest'])
                    pred.tolist()
                    #print confusion_matrix(y_test, pred)
                    #print classification_report(y_test, pred)
                    accuracy = round(accuracy_score(y_test, pred)*100,2)
                    print 'Kernel: ', k, ' - C: ', c, ' - gamma: ', g, ' --> ', accuracy
                    if accuracy > HOG[h]['accuracy']:
                        HOG[h]['accuracy'] = accuracy
                        HOG[h]['C'] = c
                        HOG[h]['gamma'] = g
                        HOG[h]['kernel'] = k
        accuracy = 0


'-----------------------------------------------------'
'------------ SVM (PCA) ------------------------------'
'-----------------------------------------------------'
#SVM calculation for each HOG + PCA

for h in HOG:
    print '------------------------', h
    for k in kernel:
        for c in C:
            for g in gamma:
                clf = sksvm.SVC(C=c, gamma = g, kernel= k, decision_function_shape='ovo', probability=True, class_weight= 'balanced')
                clf.fit(HOG[h]['PCAtrain'], y_train)
                #Predict Training
                #print 'TRAIN ', h
                pred_prob = clf.predict_proba(HOG[h]['PCAtrain'])
                pred = clf.predict(HOG[h]['PCAtrain'])
                pred.tolist()
                #print confusion_matrix(y_train, pred)
                
                #Predict Test
                #print 'TEST ', h
                pred_prob = clf.predict_proba(HOG[h]['PCAtest'])
                pred = clf.predict(HOG[h]['PCAtest'])
                pred.tolist()
                #print confusion_matrix(y_test, pred)
                #print classification_report(y_test, pred)
                accuracy = round(accuracy_score(y_test, pred)*100,2)
                print 'Kernel: ', k, ' - C: ', c, ' - gamma: ', g, ' --> ', accuracy
                if accuracy > HOG[h]['accuracy']:
                    HOG[h]['accuracy'] = accuracy
                    HOG[h]['C'] = c
                    HOG[h]['gamma'] = g
                    HOG[h]['kernel'] = k
    accuracy = 0
        
#Best parameters and accuracy for each HOG
print '---------------------------------------------------------------------'
print '6 BREEDS'
for h in HOG: print h, ' --> ', HOG[h]['accuracy'], ' *** Kernel: ', HOG[h]['kernel'] , ' *** C: ', HOG[h]['C'], ' - gamma: ', HOG[h]['gamma']
print '---------------------------------------------------------------------'

'-----------------------------------------------------'
'------------ SVM SUMMARY - NO PCA -------------------'
'-----------------------------------------------------'
#Summary (accuracy, confusion matrix and classification report for each HOG)
for h in HOG:
        print '------------------------------------------------------------------------'
        print '------------------------------------------------------------------------'
        print h, ' --> ', HOG[h]['accuracy'], ' *** C: ', HOG[h]['C'], ' - gamma: ', HOG[h]['gamma']
        clf = sksvm.SVC(C=HOG[h]['C'], gamma = HOG[h]['gamma'], kernel='rbf',
                        class_weight= 'balanced', decision_function_shape='ovo',probability=True)
        HOG[h]['SVM'] = clf.fit(HOG[h]['HOGdesTrain'], y_train)
        pred = HOG[h]['SVM'].predict(HOG[h]['HOGdesTest'])
        HOG[h]['pred_prob'] = HOG[h]['SVM'].predict_proba(HOG[h]['HOGdesTest'])
        accuracy = round(accuracy_score(y_test, pred)*100,2)
        print '------------------------------------------------------------------------'
        print confusion_matrix(y_test, pred)
        print '------------------------------------------------------------------------'
        print classification_report(y_test, pred)


'-----------------------------------------------------'
'------------ SVM SUMMARY - PCA ----------------------'
'-----------------------------------------------------'
#Summary (accuracy, confusion matrix and classification report for each HOG + PCA)

for h in HOG:
    #if h != 'HOG4':  
        print '------------------------------------------------------------------------'
        print '------------------------------------------------------------------------'
        print h, ' --> ', HOG[h]['accuracy'], ' *** C: ', HOG[h]['C'], ' - gamma: ', HOG[h]['gamma']
        clf = sksvm.SVC(C=HOG[h]['C'], gamma = HOG[h]['gamma'], kernel='rbf',
                        class_weight= 'balanced', decision_function_shape='ovo',probability=True)
        HOG[h]['SVM'] = clf.fit(HOG[h]['PCAtrain'], y_train)
        pred = HOG[h]['SVM'].predict(HOG[h]['PCAtest'])
        HOG[h]['pred_prob'] = HOG[h]['SVM'].predict_proba(HOG[h]['PCAtest'])
        accuracy = round(accuracy_score(y_test, pred)*100,2)
        print '------------------------------------------------------------------------'
        print confusion_matrix(y_test, pred)
        print '------------------------------------------------------------------------'
        print classification_report(y_test, pred)

    
print '------------------------------------------------------------------------'
print '------------------------- TEST STACKED ---------------------------------'
print '------------------------------------------------------------------------'
#Stacked predictions
finalpred = []
predictionsProb = np.zeros([len(X_test), len(breeds)])
for h in HOG:
    predictionsProb += HOG[h]['pred_prob']
sortedBreeds = sorted(breeds)
totalresults = []
for p in predictionsProb:
    max_value = max(p)
    max_index = np.where(p==max_value)
    finalpred += [sortedBreeds[max_index[0][0]]]
#Printing Stacked predictions
print '------------------------------------------------------------------------'
accuracy = round(accuracy_score(y_test, finalpred)*100,2)
print '------------------------------------------------------------------------'
print 'TEST SET: STACKED SVM --> Accuracy: ', accuracy
print '------------------------------------------------------------------------'
print confusion_matrix(y_test, finalpred), '\n'
print classification_report(y_test, finalpred)


print '------------------------------------------------------------------------'
print '-------------------------- VALIDATION ----------------------------------'
print '------------------------------------------------------------------------'
#validation without PCA
for h in HOG:
        pred = HOG[h]['SVM'].predict(HOG[h]['HOGdesVal'])
        HOG[h]['pred_probVAL'] = HOG[h]['SVM'].predict_proba(HOG[h]['HOGdesVal'])
        HOG[h]['pred_VAL'] = HOG[h]['SVM'].predict(HOG[h]['HOGdesVal'])
        accuracy = round(accuracy_score(y_val, pred)*100,2)
        print '------------------------------------------------------------------------'
        print 'VALIDATION SET: STACKED SVM: ', h, ' --> ', accuracy, '%'
        print '------------------------------------------------------------------------'
        print confusion_matrix(y_val, HOG[h]['pred_VAL']), '\n'
        print classification_report(y_val, HOG[h]['pred_VAL'])
        print '------------------------------------------------------------------------'


print '------------------------------------------------------------------------'
print '-------------------------- VALIDATION - PCA ----------------------------'
print '------------------------------------------------------------------------'
#validation with PCA (we use the results of HOG4 without PCA for this model)
for h in HOG:
    if h != 'HOG4':    
        pred = HOG[h]['SVM'].predict(HOG[h]['PCAval'])
        HOG[h]['pred_probVAL'] = HOG[h]['SVM'].predict_proba(HOG[h]['PCAval'])
        HOG[h]['pred_VAL'] = HOG[h]['SVM'].predict(HOG[h]['PCAval'])
        accuracy = round(accuracy_score(y_val, pred)*100,2)
        intconf = []
        print '------------------------------------------------------------------------'
        print 'VALIDATION SET: STACKED SVM: ', h, ' --> ', accuracy, '%'
        print '------------------------------------------------------------------------'
        print confusion_matrix(y_val, HOG[h]['pred_VAL']), '\n'
        print classification_report(y_val, HOG[h]['pred_VAL'])
        print '------------------------------------------------------------------------'


print '------------------------------------------------------------------------'
print '------------------------- VAL STACKED ----------------------------------'
print '------------------------------------------------------------------------'
finalpredVal = []
predictionsProbVal = np.zeros([len(X_val), len(breeds)])
for h in HOG:
    predictionsProbVal += HOG[h]['pred_probVAL']
totalresultsVal = []

for p in predictionsProbVal:
    max_value = max(p)
    max_index = np.where(p==max_value)
    finalpredVal += [sortedBreeds[max_index[0][0]]]

accuracy = round(accuracy_score(y_val, finalpredVal)*100,2)
print accuracy
intconf = []
for i in range(len(y_val)):
    if y_val[i] == finalpredVal[i]: intconf += [1]
    else: intconf += [0]
    
print '------------------------------------------------------------------------'
print 'VALIDATION SET: STACKED SVM --> Accuracy: ', accuracy, '%'
print '------------------------------------------------------------------------'
print confusion_matrix(y_val, finalpredVal), '\n'
print classification_report(y_val, finalpredVal)
print '------------------------------------------------------------------------'
acc, acc_min, acc_max =  hf.mean_confidence_interval(intconf, 0.95)
print '95% CI (%): (', round(acc_min*100, 2), ',', round(acc*100,2), ',', round(acc_max*100,2),')'



'#####################################################'
'################## RANDOM FORESTS ###################'
'#####################################################'

'-----------------------------------------------------'
'------------ RANDOM FOREST - 6 BREEDS --------------'
'-----------------------------------------------------'

#WITHOUT PCA
for h in HOG:
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    HOG[h]['RF'] =  clf.fit(HOG[h]['HOGdesTrain'], y_train)
    pred = HOG[h]['RF'].predict(HOG[h]['HOGdesVal'])
    HOG[h]['pred_prob'] = HOG[h]['RF'].predict_proba(HOG[h]['HOGdesVal'])
    accuracy = round(accuracy_score(y_val, pred)*100,2)
    print '------------------------------------------------------------------------'
    print 'RANDOM FOREST: ', h, ' --> ', accuracy, '%'
    print '------------------------------------------------------------------------'
    print confusion_matrix(y_val, pred), '\n'
    print classification_report(y_val, pred)
    print '------------------------------------------------------------------------'

#WITH PCA
for h in HOG:
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    HOG[h]['PCARF'] =  clf.fit(HOG[h]['PCAtrain'], y_train)
    pred = HOG[h]['PCARF'].predict(HOG[h]['PCAval'])
    HOG[h]['pred_probPCARF'] = HOG[h]['PCARF'].predict_proba(HOG[h]['PCAval'])
    accuracy = round(accuracy_score(y_val, pred)*100,2)
    print '------------------------------------------------------------------------'
    print 'RANDOM FOREST WITH PCA: ', h, ' --> ', accuracy, '%'
    print '------------------------------------------------------------------------'
    print confusion_matrix(y_val, pred), '\n'
    print classification_report(y_val, pred)
    print '------------------------------------------------------------------------'
        
print '------------------------------------------------------------------------'
print '------------------------- RF STACKED -----------------------------------'
print '------------------------------------------------------------------------'
#WITHOUT PCA
finalpredVal = []
predictionsProbVal = np.zeros([len(X_val), len(breeds)])
for h in HOG:
    predictionsProbVal += HOG[h]['pred_prob']
totalresultsVal = []

for p in predictionsProbVal:
    max_value = max(p)
    max_index = np.where(p==max_value)
    finalpredVal += [sortedBreeds[max_index[0][0]]]

accuracy = round(accuracy_score(y_val, finalpredVal)*100,2)
print accuracy
intconf = []
for i in range(len(y_val)):
    if y_val[i] == finalpredVal[i]: intconf += [1]
    else: intconf += [0]

#WITH PCA

finalpredValPCA = []
predictionsProbValPCA = np.zeros([len(X_val), len(breeds)])
for h in HOG:
    predictionsProbValPCA += HOG[h]['pred_probPCARF']
totalresultsValPCA = []

for p in predictionsProbValPCA:
    max_value = max(p)
    max_index = np.where(p==max_value)
    finalpredValPCA += [sortedBreeds[max_index[0][0]]]

accuracy = round(accuracy_score(y_val, finalpredValPCA)*100,2)
print accuracy
intconf = []
for i in range(len(y_val)):
    if y_val[i] == finalpredValPCA[i]: intconf += [1]
    else: intconf += [0]


#WITHOUT PCA
print '------------------------------------------------------------------------'
print 'VALIDATION SET: STACKED RF --> Accuracy: ', accuracy, '%'
print '------------------------------------------------------------------------'
print confusion_matrix(y_val, finalpredVal), '\n'
print classification_report(y_val, finalpredVal)
print '------------------------------------------------------------------------'
acc, acc_min, acc_max =  hf.mean_confidence_interval(intconf, 0.95)
print '95% CI (%): (', round(acc_min*100, 2), ',', round(acc*100,2), ',', round(acc_max*100,2),')'

#WITH PCA
print '------------------------------------------------------------------------'
print 'VALIDATION SET: STACKED RF (PCA) --> Accuracy: ', accuracy, '%'
print '------------------------------------------------------------------------'
print confusion_matrix(y_val, finalpredValPCA), '\n'
print classification_report(y_val, finalpredValPCA)
print '------------------------------------------------------------------------'
acc, acc_min, acc_max =  hf.mean_confidence_interval(intconf, 0.95)
print '95% CI (%): (', round(acc_min*100, 2), ',', round(acc*100,2), ',', round(acc_max*100,2),')'
