

# Deeplearning4J
This is a java deep learning project repository with DL4J based projects made from scratch. Make sure to switch over pom.xml changes as per your convenience whether you are using GPU or not. 

![pom.xml changes](https://user-images.githubusercontent.com/517415/41832175-8cd327a8-7868-11e8-82cd-05cc429d010a.png)

Projects included so far:

## 1) [Customer loss prediction using standard feed forward network]
(https://github.com/rahul-raj/Deeplearning4J/blob/master/src/main/java/CustomerLossPrediction.java)

 [![DL4J - Customer Loss prediction example](https://img.youtube.com/vi/DGllOCWL5w0/0.jpg)](https://www.youtube.com/watch?v=DGllOCWL5w0)
 
Given **n** labels defined, obtain the probability of a customer who leaves the bank. The problem statement is taken from a course driven by superdatasciene team. They discuss Keras implementation for the same while this is a small attempt to do the same with Java. Keras has managed to obtain 83% consistent accuracy however DL4J model outperform Keras as it gives 85.5% of consistent accuracy. 

![GPU changes](https://user-images.githubusercontent.com/517415/41832945-f24ea744-786b-11e8-8b99-022508c2e10a.png)

## 2) [Animal classification using CNN (no pre-built model)]
(https://github.com/rahul-raj/Deeplearning4J/blob/master/src/main/java/AnimalClassifier.java)

You may observe over-fitting if you're using GPU or running CPU with about 100 epochs. 30-60 epochs should be fine for a CPU execution, however feel free to fork this up, experiment on your own and send me a pull request if you obtain a good CNN model better than this existing one! DL4J github examples used alexnet while we just coded a custom model from scratch. 4 animal labels and around 6000 images in total (training+testing). More images would result in an obvious increase of accuracy.

## 3) [HyperParameter tuning using Java]
(https://github.com/rahul-raj/Deeplearning4J/blob/master/src/main/java/HyperParamTuning.java)

![Make sure to implement your own data provider](https://user-images.githubusercontent.com/517415/41833939-f8f4665c-786f-11e8-917a-0f8fd97851a1.png)

A bug has been reported to DL4J while I was coding this one, however they made a fix and it's available only on snapshots. Consider this as unstable version now until it's tested completely. Currently I'm checking with DL4J team to identify possible pitfall with the paramater space config or a possible bug that will soon to be reported to them. 

## 4) [Santander Value Prediction Kaggle Challenge (Data pre-processing so far)]
(https://github.com/rahul-raj/Deeplearning4J/blob/master/src/main/java/SantanderValuePrediction.java)

Probably the biggest learning curve where I plan to implement this Kaggle challenge all by using Java. Objective is to perform all the tasks (data pre-processing, transform, load, network config and evaluation) using Java. We're trying to build a production-ready real-time deep learning application. 

