# Covid-Detection

Coronavirus disease (COVID-2019) is emerged in Wuhan, China in 2019. It has spreadthroughout the world from the year 2020. Millions of peoples were affected and caused death to
them. To avoid this spreading of COVID-2019 various precautions and restrictions have beentaken by all the nations. At the same time infected persons are need to identified, isolate andmedical treatment should be provided to them. Due to deficient number of Reverse Transcription Polymerase Chain Reaction (RT-PCR) tests, Chest Computed tomography (CT) has becomes effective technique to diagnosis of COVID-19. 

VGG 19 is used for automatic detection of COVID-19. The discriminative features from last layer of these architectures were feed as an input to suppor tvector machine (SVM) for the prediction of COVID-19. The different hyper parameters of the SVM were optimized using differnt optimization algorithm. The experimental analyses are conducted to identify the performance of proposed CNN architecture using COVID-19 radiology database. The entire database was separated by 70% training and 30% testing respectively. The dataset consists of 453 COVID-19 images and 497 non-COVID images. The model achieved an accuracy of 90%. The proposed models can be used to assist radiologists and physicians to reduce the misdiagnosis rates and to validate the positive COVID -19 infection cases.

FLOWCHART:


![image](https://user-images.githubusercontent.com/56121394/125592058-d61412ed-c913-452b-a217-5bdc346b6216.png)

STATE OF ART:

In recent years, Convolutional Neural Networks (CNNs) have become the state-of-the-art for object recognition in computer vision. Typically, a CNN consists of several convolutional layers, followed by two fully-connected layers. An intuition behind this is that the convolutional layers learn a better representation of the input data, and the fully connected layers then learn to classify this representation based into a set of labels.

However, before CNNs started to dominate, Support Vector Machines (SVMs) were the state-of-the-art. So, it seems sensible to say that an SVM is still a stronger classifier than a two-layer fully-connected neural network

So, we have used SVM as the final layer for Classification.

![image](https://user-images.githubusercontent.com/56121394/125592139-2bb980ef-192c-44b8-8dc8-0db9e1f96a13.png)

