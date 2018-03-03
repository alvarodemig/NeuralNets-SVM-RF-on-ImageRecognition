# SVM vs RF vs NeuralNets on fine grain Image Recognition

Dataset: https://www.kaggle.com/c/dog-breed-identification/data 

Data and Preliminary analysis:
  - Image Transformation:
    - Size normalization
    - Color datasets: colored, black & white, grey scale
   - Data Augmentation: horizontal flip, random sharpen, lightness, contrast, affination, RGB modification...
   - Histogram of Oriented Gradients (HOG): feature extractor to be used in SVM and MLP
   - Principal Component Analysis: dimensionality reduction (combined with HOG)
   
Model Building and Validation:
  - HOG + Support Vector Machine (with and without PCA) and stacked Models
  - HOG + Random Forests (with and without PCA) and stacked Models
  - HOG + Multilayer Perceptron and Stacked Models (minimum loss and max accuracy)
  - Convolutional Neural Network and Stacked Models (minimum loss and max accuracy)
    
Not using pretrained Neural Networks
-------------------------------------------------------------------------------------------

Files:
  - SVMRFbreeds: MAIN FILE. SVM and RF algorithms. Python 2.7
  - CNNbreeds: Convolutional Neural Net. Python 3.5
  - MLPbreeds: MultiLayer Perceptron. Python 3.5
  - breedsHelpFunctions2: help functions. Compatible with Python 2.7 and 3.5
