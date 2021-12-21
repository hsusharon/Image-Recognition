This project contains four differnt sections.(Detailed information in the report)
1.SVM
2.Logistic Regression
3.Convolutional Neural Network
4. Transfer Learning
Dataset for the first three section is the MNist hand written digits.
Dataset for transfer learning is the 10-monkey-species from kaggle

The complete file is in the link below(containing datasets):
https://drive.google.com/file/d/1fB2QUdiWf7rFadIP6pPRQSQnaXvvA9aa/view?usp=sharing

There are two different types of code in the code folder(Python, Jupyter Notebook), running on Jupyter notebook is recommended
There are 4 ipynb files in the /code/ipynb folder that runs each section in the project
package required to run the code:
tensorflow (version 2.7.0), panda, numpy, sklearn, os, cv2

1.633Proj2_SVM.py runs the SVM section in the project
  The experiment is tested in three different kernels 
  run time approximately 10 minutes(with GPU)

2.633Proj2_LR.py runs the Logistic Regression section in the project
 
3.633Proj2_CNN.py runs the CNN(mnist dataset) of the section
    
4.633Proj2_TL.py runs the Transfer Learning section in the project
  The training dataset should be place in the route of "C:/Users/USER/Desktop/633 Proj2/data/Monkey Database/training/training/"
  The testing dataset should be place in the route of "C:/Users/USER/Desktop/633 Proj2/data/Monkey Database/validation/validation/"
  There are two models, one trains in 25 epoch and the other trains in 50 epoch
