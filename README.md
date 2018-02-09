# multi2-instance-learning
Multi-multi instance learning for captcha-like image recognition. This is my final project for a Machine Learning exam.

See http://ecmlpkdd2017.ijs.si/papers/paperID248.pdf

Instructions:
1) Download MNIST dataset (http://yann.lecun.com/exdb/mnist/) and put the files in the project folder. 
2) Generate some data; for example, launch python:

    [1] from data_generation import *
    [2] trainx, trainy = create_dataset(0, 80000, 5, image_size=(80,80), data_type='training')
    [3] testx, testy = create_dataset(0, 8000, 5, image_size=(80,80), data_type='testing')
    [4] store_datasets(trainx, trainy, testx, testy)

3) Train the model typing 'python train.py' 
   Feel free to tweak some parameters inside the code, it should be enough self-explanatory.
   
That's all folks.
    
    
