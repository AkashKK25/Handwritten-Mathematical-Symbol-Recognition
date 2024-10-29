# Final Project

<!-- ABOUT THE PROJECT -->
## About The Project


This project implements a machine learning system for Handwritten Mathematical Symbols classification. 


The dataset used for training is a collection of handwritten mathematical symbols by the students in the course EEL5840 - Fundamental of Machine Learning course requirements at the University of Florida.
The training dataset has *9032* images and available in the form of *numpy arrays*. 

Each image of one hand-written symbol is represented by 300*300 pixels and all the images are combined into a numpy array named data_train.npy. 
Target classes of each image containing a symbol is integer-encoded and combined into another numpy array t_train.npy.

The data for each instance is stored in columns in the 'data_train.npy' file.

These datasets are used for training our classification model.

The training dataset contains the symbols from integer-encoded 10 classes as follows:   
0: x   
1: square root    
2: plus sign   
3: negative sign   
4: equal   
5: percent   
6: partial   
7: product   
8: pi   
9: summation   

The implementation details of the project can be referred in the project report.


<!-- GETTING STARTED -->
## Getting Started

The project contain following files:

1. [train.ipynb](https://github.com/UF-EEL5840-F22/final-project---code-report-the-optimizers/blob/main/train.ipynb)
   This is the code file for training the classifier.
   
2. [test.ipynb](https://github.com/UF-EEL5840-F22/final-project---code-report-the-optimizers/blob/main/test.ipynb)
   This is the code file for testing the classifier.
   
3. [the_optimizers_classification_model.h5](https://uflorida-my.sharepoint.com/:u:/g/personal/akash_kondaparth_ufl_edu/EQV32fQNuAlFmWwwRD-1Sb8BBrzsaR3Y8cSxGDv72Ta-gA?e=cM1rYm)
   This is the saved Trained Model which is used in test.ipynb
    
4. [Project Report](https://github.com/UF-EEL5840-F22/final-project---code-report-the-optimizers/blob/main/Project Report.pdf)
   This report describes detail implemetation of the project


### Dependencies

This project is implemeted using Python language. Open-source Python libraries are used in this project.
The code is compatible with UF HiPerGator TensorFlow-2.7.0 kernel. 

Libraries are:
1. TensorFlow
2. OpenCV
3. Numpy
4. Scikit-learn
5. Matplotlib


<!-- USAGE EXAMPLES -->
## Usage

1. Training the model:

    train.ipynb is to be executed to train the classifier.
   
    Input: data_train.npy, t_train.npy
    
    Output: The trained model is saved in the_optimizier_classification_model.h5
    This saved model can be used directly for testing any handwritten-mathematical symbol.
    
    
    To Run the Training Model:
    
    1. Please modify following variables in cell 2:
    
    basepath = "./"
    
    training_data_filename = 'data_train.npy'
    
    training_target_filename = 't_train.npy'
    
    2. Run all the cells of the train.ipynb file
   
2. Use the model for testing and predicting new data

   test.ipynb is to be executed to test Handwritten mathematical symbols.
   Test function will predict the class of each test symbol using the trained model. 
   If the symbol does not belong to any of the 10 classes then it will be predicted as an unknown symbol (integer value: 10)
      
   ** Note: For testing/prediction re-training is not required **
   
   Input: data_test.npy should be in similar format of data_train.npy i.e. each symbol is represented by 300 * 300 pixels.
   Output: Predicted labels vector
           If targets are provided in t_test.npy, then model would provide the accuracy,
           confusion matrix and classification report
    
    
    To Run the Test Model:
    
    1. Please modify following variables in cell 2:
    
    basepath = "./"
    
    test_data_filename = 'X_test.npy'
    
    test_label_filename = 't_test.npy'
    
    2. Run all the cells of the test.ipynb file    
    
Model Architecture:
Model is derived from ResNet50 model, with imagenet trained weights through transfer learning. The model architecture is as shown in the figure below:
![Alt text](https://github.com/UF-EEL5840-F22/final-project---code-report-the-optimizers/blob/main/images/ResNetmodelfig.png)

The Resulting training curves:
![Alt text](https://github.com/UF-EEL5840-F22/final-project---code-report-the-optimizers/blob/main/images/training_curves.png)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

For this project implemetation we are thankful to our course instructor Dr. Catia Silva and University of Florida.

* [Catia Silva](https://faculty.eng.ufl.edu/catia-silva/)
* [UF HiPerGator](https://ood.rc.ufl.edu/)