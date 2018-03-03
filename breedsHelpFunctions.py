import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import cv2, os, scipy, random, math, csv, keras
from sklearn.decomposition import PCA
from imgaug import augmenters as iaa


#Create new pictures with the chosen transformations and cropping square size. Example:
'''
pathpics = getcwd() + '/trainKaggle/'
pictures = [f for f in listdir(pathpics) if isfile(join(pathpics, f))]
picTransfCreation(pathpics, pictures, 'no', 100, ['trans'])
'''
def picTransfCreation(pathpics, picturesListDir, yesnocrop, sqsize = 100, transformations = ['crop', 'gray', 'bw', 'gaussianblur', 'gaussiannoise', 'trans']):
    #pathpics: path where the pics are
    #picturesListDir: list of files to be processed
    #do you want to crop? 'yes'or 'no'
    #sqsize: square size of the cropping
    #transformations: [list to include processed wanted] bw (black&white), crop, gray [list]
    
    basepath = os.getcwd()
    counter = 0
    
    c = 'crop' in transformations
    g = 'gray' in transformations
    b = 'bw' in transformations
    gb = 'gaussianblur' in transformations
    gn = 'gaussiannoise' in transformations
    trans = 'trans' in transformations
    if g:
        croppath = basepath + '/CROPPED' + str(sqsize) + '/'
        if not os.path.exists(croppath):
            os.makedirs(croppath)
    if g:
        graypath = basepath + '/GRAY' + str(sqsize) + '/'
        if not os.path.exists(graypath):
            os.makedirs(graypath)
    if b:
        bwpath = basepath + '/BW' + str(sqsize) + '/'
        if not os.path.exists(bwpath):
            os.makedirs(bwpath)
    if gb:
        gbpath = basepath + '/GB' + str(sqsize) + '/'
        if not os.path.exists(gbpath):
            os.makedirs(gbpath)
    if gn:
        gnpath = basepath + '/GN' + str(sqsize) + '/'
        if not os.path.exists(gnpath):
            os.makedirs(gnpath)

    if trans:
        transpath = basepath + '/TRANS' + str(sqsize) + '/'
        if not os.path.exists(transpath):
            os.makedirs(transpath)  
    
    for pic in picturesListDir:
        counter += 1    
        im = Image.open(pathpics + pic)
        width, height = im.size   # Get dimensions
        if yesnocrop == 'yes':
            if width > height:
                newsize = math.ceil(float(sqsize)*width/height), sqsize
                h = int(math.floor((newsize[0] - newsize[1])/2))
                h1 = newsize[0] - 2*h - sqsize
                im.thumbnail(newsize, Image.ANTIALIAS)
                im = im.crop((h,0,newsize[0]-h-h1,newsize[1]))
            else:
                newsize = sqsize, math.ceil(float(sqsize)*height/width)
                h = int(math.floor((newsize[1] - newsize[0]))/2)
                h1 = newsize[1] - 2*h - sqsize
                im.thumbnail(newsize, Image.ANTIALIAS)
                im = im.crop((0,h,newsize[0],newsize[1]-h-h1))

        imsplit = pic.split('_')[0].split('.')[0]
        if c:
            imcrop = croppath + imsplit + '_CR' + str(sqsize) + '.jpeg'
            im.save(imcrop, format=im.format)
            print(counter, imsplit,' CROPPED CREATED')

        if g:
            gray = im.convert('L')  
            imgray = graypath + imsplit + '_GR' + str(sqsize) + '.jpeg'
            gray.save(imgray)
            print(counter, imsplit,' GREY SCALE CREATED')

        if b:
            bw = im.convert('1')  
            imbw = bwpath + imsplit + '_BW' + str(sqsize) + '.jpeg'
            bw.save(imbw)
            print(counter, imsplit,' BLACK AND WHITE CREATED')

        if gb:
            imarray = np.array(im)
            seqGB = iaa.Sequential([iaa.GaussianBlur(2)])
            imgb = gbpath + imsplit + '_GB' + str(sqsize) + '.jpeg'
            photo = seqGB.draw_grid(imarray, cols=1, rows=1)
            scipy.misc.imsave(imgb, photo)
            print(counter, imsplit,' GAUSSIAN BLUR CREATED')
            
        if gn:
            imarray = np.array(im)
            seqGN = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=[10,10])])
            imgn = gnpath + imsplit + '_GN' + str(sqsize) + '.jpeg'
            photo = seqGN.draw_grid(imarray, cols=1, rows=1)
            scipy.misc.imsave(imgn, photo)
            print(counter, imsplit,' GAUSSIAN NOISE CREATED')
            
        if trans:
            TRANSseq = iaa.Sequential([
                iaa.Fliplr(1), # horizontal flips
                # Strengthen or weaken the contrast in each image.
                iaa.ContrastNormalization(random.uniform(0.80, 1.25)),
                # Make some images brighter and some darker.
                # In 20% of all cases, we sample the multiplier once per channel,
                # which can end up changing the color of the images.
                iaa.Multiply(random.uniform(0.9, 1.1), per_channel=0.1),
                # Apply affine transformations to each image.
                # Scale/zoom them, translate/move them, rotate them and shear them.
                iaa.Affine(
                    scale={"x": random.uniform(1, 1.15), "y": random.uniform(1, 1.15)},
                    rotate = random.randint(-15, 15)),
                #Sharpen images
                iaa.Sharpen(alpha = random.uniform(0.0, 0.3), lightness = random.uniform(0.8, 1.25)),
                #Add values to pixel
                iaa.Add(random.randint(-20, 20))
                ], random_order=True) # apply augmenters in random order
                         
            imarray = np.array(im)  
            imtrans = transpath + imsplit + '_TRANS' + str(sqsize) + '.jpeg'
            photo = TRANSseq.draw_grid(imarray, cols=1, rows=1)
            scipy.misc.imsave(imtrans, photo)
            print(counter, imsplit,' TRANS CREATED')


#Retrieve the label (breed) of each picture
def retrieveLabels():
    breedDict = {}
    dogDict = {}
    path = os.getcwd() + '/labels.csv'
    open(path)
    with open(path, 'rb') as csvfile:
        labelscsv = csv.reader(csvfile, delimiter=',')
        for row in labelscsv:
            dogDict[row[0]] = row[1]
            if row[1] in breedDict:
                breedDict[row[1]]['pics'] += [row[0]]
                breedDict[row[1]]['count']+= 1
            else:
                breedDict[row[1]] = {}
                breedDict[row[1]]['pics'] = [row[0]]
                breedDict[row[1]]['count'] = 1
    return breedDict, dogDict

#Create Dummy Variables from a list of classes and return the classes and dummy variables sorted
def create_dummies(varList):
    classes = sorted(set(varList))
    numvars = len(classes)
    dummy = np.zeros([len(varList), numvars], float)
    var = {}
    c = 0
    for v in classes:
        var[v] = c
        c +=1
    for v in range(len(varList)):
        index = var[varList[v]]
        dummy[v][index] = 1
    return classes, dummy

#Create a Matrix from a pic file to use as input in NN
def picToMatrix(picRoot):
    im = Image.open(picRoot)
    imList = np.array(im)
    try:
        x,y = imList.shape
        deep = 1
    except:
        x,y, deep = imList.shape
    return imList.reshape(x,y,deep)

	
#Transform pictures to arrays and create a INPUT and TARGET frames (For SVM code)
def picToArrayDF(picFolder, targetPicnamesList, width, height):
    dataset = np.empty((len(targetPicnamesList), width*height))
    counter = 0
    target = []
    for pic in targetPicnamesList:
        im = Image.open(picFolder + pic[1])
        picarray = np.array(im)
        picarray = picarray.reshape(width * height)
        dataset[counter] = picarray
        target += [pic[0]]
        counter += 1
        print(counter)
    return dataset, target
	
	
#Create a numpy matrix from a List of numbers to use as input in NN
def listToMatrix(mylist):
    l = np.array(mylist)
    x,y = l.shape
    r = l.reshape(x,y,1)
    return r

#dA
def MatrixToDataset(dataset, width, height, color):
    dlen = len(dataset)
    dataset_pic = np.empty((dlen, width, height, color))
    counter = 0
    for p in range(dlen):
        counter +=1
        dataset_pic[p] = picToMatrix(dataset[p])/255
        if counter % 50 == 0 : print(counter, ' / ', dlen)
    return dataset_pic

#Predict a model and return accuracy, confusion matrix and classification report
def modelEvaluation(model, X, Y, classes):
    pred_prob = model.predict(X, verbose = 0)
    pred = []
    for p in pred_prob:
        max_value = max(p)
        max_index = np.where(p==max_value)
        pred += [classes[max_index[0][0]]]
        
    accuracy = round(accuracy_score(Y, pred)*100,2)
    print('-----------------------------\n')
    print('Accuracy: ', accuracy, '%\n')
    print(confusion_matrix(Y, pred))
    print(classification_report(Y, pred))
    print('-----------------------------')
    return pred, accuracy, pred_prob

#Add extra data (augmented) to the train or change the validation set (grey, blurred... pictures)
def datasetExpansion(traintxt, testtxt, valtxt, extratrain, validsets):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X_val = []
    y_val = []
    
    f = open(traintxt, "r" )
    for line in f:
        splits = line.split(' --- ')
        X_train += [splits[0]]
        y_train += [splits[1].replace('\n','')]
    f.close()
    
    f = open(testtxt, "r" )
    for line in f:
        splits = line.split(' --- ')
        X_test += [splits[0]]
        y_test += [splits[1].replace('\n','')]
    f.close()
    
    f = open(valtxt, "r" )
    for line in f:
        splits = line.split(' --- ')
        X_val += [splits[0]]
        y_val += [splits[1].replace('\n','')]
    f.close()

    if type(extratrain) is list:
        print('Extra dataset train')
        for e in extratrain:
            folder = e[0]
            filename = e[1]
            for i in range(len(X_train)):
                X_train += [X_train[i].replace('CR100', folder).replace('CROPPED100', filename)]
                y_train += [y_train[i]]

    if type(validsets) is list:
        print('Extra dataset validation')
        X_testNew = X_test[:]
        X_valNew = X_val[:]
        y_testNew = y_test[:]
        y_valNew = y_val[:]
        for e in validsets:
            folder = e[0]
            filename = e[1]
            for i in range(len(X_test)):
                X_test += [X_test[i].replace('CR100', folder).replace('CROPPED100', filename)]
                y_test += [y_test[i]]
            for i in range(len(X_val)):
                X_val += [X_val[i].replace('CR100', folder).replace('CROPPED100', filename)]
                y_val += [y_val[i]]

        X_test = X_testNew[:]
        X_val = X_valNew[:]
        y_test = y_testNew[:]
        y_val = y_valNew[:]
    print('Dataset path loaded')
    return X_train, X_test, X_val, y_train, y_test, y_val

#Hog Descriptor calculator for a picture
def hogDescriptorCalculation(grayPic, winSize = (100,100), cellSize = (10,10), blockSize =(20,20), blockStride = (10,10), nbins = 9):
    #HOG parameters
    # winSize -> picture size
    # celSize -> Size of features
    # blockSize -> Usually cellsize * 2 --> illumination
    # blockStride -> Overlap between neighboring blocks. Usually 50% of blockSize
    # nbins -> Gradients between 0 and 180 degrees
    
    #Default values. Do not change unless you know HOG descriptor very well
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True
    
    #Hog Descriptor
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)
    return hog.compute(grayPic)

#PCA calculator from the HOG list, and print the explained variability
def PCAhog(X, components):
    pca = PCA(n_components=components, svd_solver='arpack')
    pcafit = pca.fit(X)
    explainedratio = pca.explained_variance_ratio_
    varsum = 0
    for i in explainedratio: varsum += i
    print(varsum)
    #print(pca.singular_values_)  
    return pcafit
	

#We will use this class to store the values of the accuracy and loss in each epoch
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
        
    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

#Calculate the confidence intervals
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h
