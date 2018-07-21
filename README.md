
# Deeplearning4J
A java deep learning repository with DL4J based projects made from scratch. Projects included so far:

1. Customer loss prediction using standard feed forward network
2. Animal classification using CNN
3. HyperParameter tuning using Java
4. Santander Value Prediction Kaggle Challenge

Make sure to switch over pom.xml changes as per your convenience whether you are using a GPU or not. 

![pom.xml changes](https://user-images.githubusercontent.com/517415/41832175-8cd327a8-7868-11e8-82cd-05cc429d010a.png)

## 1. Customer loss prediction using standard feed forward network

Given defined **n** labels, obtain the probability of a customer who leaves the bank. The problem statement is taken from a course driven by a superdatasciene team. They discuss a solution using Keras implementation while this is an attempt to do the same with Java. The DLJ4 model gives a consistent 85.5% accuracy which is better than the Keras model of 83% accuracy.

File link: https://github.com/rahul-raj/Deeplearning4J/blob/master/src/main/java/examples.CustomerLossPrediction.java

Code execution:

 [![DL4J - Customer Loss prediction example](https://img.youtube.com/vi/DGllOCWL5w0/0.jpg)](https://www.youtube.com/watch?v=DGllOCWL5w0)

![Optimization](https://user-images.githubusercontent.com/517415/41850738-4db23466-78a3-11e8-802d-c89df35a227b.png)

## 2. Animal classification using CNN (no pre-built model)

You may observe over-fitting if you're using GPU or running CPU with about 100 epochs. 30-60 epochs should be fine for a CPU execution, however feel free to fork this up, experiment on your own and send me a pull request if you obtain a good CNN model better than this existing one! DL4J github examples used alexnet while we just coded a custom model from scratch. 4 animal labels with 6000 images in total (training+testing). More images would result in an obvious increase of accuracy.

File link: https://github.com/rahul-raj/Deeplearning4J/blob/master/src/main/java/examples.AnimalClassifier.java

**Update:** GPU fix has been done by DL4J team and updates available only on snapshots since changes are on ND4j backend. Yet to test them.

## 3. HyperParameter tuning using Java

![Make sure to implement your own data provider](https://user-images.githubusercontent.com/517415/41833939-f8f4665c-786f-11e8-917a-0f8fd97851a1.png)

A bug has been reported to DL4J while I was coding this one, however they made a fix and it's available only on snapshots. Consider this as unstable version now until it's tested completely. Currently I'm checking with DL4J team to identify possible pitfall with the paramater space config or a possible bug that will soon to be reported to them. 

File link: https://github.com/rahul-raj/Deeplearning4J/blob/master/src/main/java/examples.HyperParamTuning.java

**Update:** Code fix has been done. Until a new version of DL4J is released, change your pom.xml to use snapshot version of DL4J or just copy the `examples.DataSetIteratorSplitter` directly from the DL4J master. Note that I have already made a local copy of this class into the project repository.

Code execution:

[![DL4J - HyperParameter tuning](https://img.youtube.com/vi/tg6t7LMdMow/0.jpg)](https://www.youtube.com/watch?v=tg6t7LMdMow)

## 4. Santander Value Prediction Kaggle Challenge (Data pre-processing so far)

Probably the biggest learning curve where I plan to implement this Kaggle challenge all by using Java. Objective is to perform all the tasks (data pre-processing, transform, load, network config and evaluation) using Java. We're trying to build a production-ready real-time deep learning application. 

File link: https://github.com/rahul-raj/Deeplearning4J/blob/master/src/main/java/examples.SantanderValuePrediction.java

