# spotify-hit-predictor
This is a juptyer notbook file with an SVM model that will predict the commerical success of a spotify song. 


Can Supervised Learning Predict Commercial Recognition from The Acoustic Attributes of a Song?

Introduction

The connection between how art in musical composition is expressed and how well it's recognized in the business world is a common topic of discussion. Can the specific acoustic qualities of a song really tell us if it will be successful, considering things like how popular it gets, its cultural impact, and overall influence? By examining the connection between these sound qualities and a song's success, we can learn valuable things about how art and recognition are able to relate to each other. By constructing an accurate model to predict commercial recognition, based on acoustic attributes of a song, we believe supervised machine learning may be able to extrapolate a common zeitgeist, among people.
Dataset

A dataset was crowdsourced from Kaggle, titled The Spotify Hit Predictor Dataset (1960-2019). (Link: https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset). As a major music streaming platform, Spotify offers a wealth of song data, including various acoustic characteristics.

The dataset in question encompasses features for tracks fetched using Spotify's Web API. The dataset contains 41 106 songs. Each song has 17 features or dimensions associated with it. The data was processed to have 17 dimensions, all of which containing an aspect of an acoustic attribute of the song, except for the decade attribute, and the class attribute "target". The class attribute which labels if the song was a commercial success or not, is a symmetric binary type (i.e, True or False). Commercial success, meaning that the song has appeared at least once on the weekly list of top 100 tracks, as issued by Billboard magazine, in its respective decade.

Thus, each track is labeled either '1' (Hit) or '0' (Flop) based on specific criteria determined by the dataset's author. (It's important to note that the term "Flop" does not imply that the track is of inferior quality but rather that it may not have gained mainstream popularity).

The supervised learning models will be binary classifiers. This class attribute is a label which will be used during supervised learning, that will define the relationship to predict commercial success, based on all the other attributes.

Int64Index: 41106 entries, 0 to 6397

Data columns (total 17 columns):
Column Non-Null Count Dtype

0 danceability 41106 non-null float64

1 energy 41106 non-null float64

2 key 41106 non-null int64

3 loudness 41106 non-null float64

4 mode 41106 non-null int64

5 speechiness 41106 non-null float64

6 acousticness 41106 non-null float64

7 instrumentalness 41106 non-null float64

8 liveness 41106 non-null float64

9 valence 41106 non-null float64

10 tempo 41106 non-null float64

11 duration_ms 41106 non-null int64

12 time_signature 41106 non-null int64

13 chorus_hit 41106 non-null float64

14 sections 41106 non-null int64

15 target 41106 non-null int64

16 decade 41106 non-null int64

dtypes: float64(10), int64(7)

Figure A: List of all attributes and data types.

The non-numerical dimensions, 'track', 'artist', and 'uri' were removed during processing. These attributes were dropped from the processed data, due to being nominal values. Therefore, the remaining 17 dimensions are all numerical, being integers, or floating-point numbers, as seen in Figure A.

speechiness acousticness instrumentalness liveness \

count 41106.000000 41106.000000 41106.000000 41106.000000

mean 0.072960 0.364197 0.154416 0.201535

std 0.086112 0.338913 0.303530 0.172959

min 0.000000 0.000000 0.000000 0.013000

25% 0.033700 0.039400 0.000000 0.094000

50% 0.043400 0.258000 0.000120 0.132000

75% 0.069800 0.676000 0.061250 0.261000

max 0.960000 0.996000 1.000000 0.999000

Figure B.

valence tempo duration_ms time_signature chorus_hit \

count 41106.000000 41106.000000 4.110600e+04 41106.000000 41106.000000

mean 0.542440 119.338249 2.348776e+05 3.893689 40.106041

std 0.267329 29.098845 1.189674e+05 0.423073 19.005515

min 0.000000 0.000000 1.516800e+04 0.000000 0.000000

25% 0.330000 97.397000 1.729278e+05 4.000000 27.599792

50% 0.558000 117.565000 2.179070e+05 4.000000 35.850795

75% 0.768000 136.494000 2.667730e+05 4.000000 47.625615

max 0.996000 241.423000 4.170227e+06 5.000000 433.182000

Figure C.

sections target decade

count 41106.000000 41106.000000 41106.000000

mean 10.475673 0.500000 52.925607

std 4.871850 0.500006 32.562672

min 0.000000 0.000000 0.000000

25% 8.000000 0.000000 10.000000

50% 10.000000 0.500000 60.000000

75% 12.000000 1.000000 80.000000

max 169.000000 1.000000 90.000000

Figure D.

As seen in Figures B, C, D, the attributes of speechiness, acousticness, instrumentalness, liveness and valence, are all interval-scaled, as the values range from 0.0 to 1.0.

_ Data Preprocessing _

Given the nature of the dataset and the requirements of the models to be employed, the following preprocessing steps are taken:

    Categorical Variable Removal: Any categorical variables present in the dataset are removed to ensure compatibility with the chosen models.
    Standard Scaling: The data is subjected to standard scaling to ensure that all features have a mean of 0 and a standard deviation of 1. This step is crucial to ensure that the models, especially SVM, function optimally.
    Data Splitting: The dataset is split into training, validation, and test sets in a 70%-15%-15% ratio. This division ensures that the models are trained on a comprehensive dataset, validated to avoid overfitting, and finally tested on unseen data to evaluate performance.

Algorithms Used

    Support Vector Machine (RBF Kernel)

After examining the dataset, it was determined that an algorithm which engaged in supervised learning and that can scale with high dimensionality, would be most optimal. Support Vector Machines (SVM) are supervised learning models that aim to calculate the margins of a hyperplane to classify whether a datapoint is in two different classes.[1] A hyperplane is an area that maximizes the margin or distance between points in separate classes, which draws the line of classification, often called the decision boundary. [2] The Support Vectors are the points that lie in the hyperplane, providing the best margin for classifying between separate classes. [3] SVMs can compute multidimensional classifications by using kernel tricks that can project inputs into many layers or dimensions, which allow for comparison of many features. [4] For this dataset and problem definition, a radial basis function (RBF) kernel was selected since it follows a (sum) of a Gaussian Distribution on the support vectors [5], which makes it easy for projecting data onto higher dimensions, which is much needed for a dataset of this dimensionality.

    AdaBoost

An ensemble method that combines the outputs of multiple weak learners to produce a strong learner. It focuses on instances that are harder to classify, making it adaptive in nature.

We have chosen to use decision tree classifiers as weak classifiers. To implement this algorithm, sample weights need to be defined. The sample weight of each sample in the database is 1/n at initialization (n is the total number of samples).

After that, add a loop with a number of cycles of "n_estimators" (the number of weak classifiers specified by the user), and the loop performs the following three operations:

    Train a weak classifier and calculate its error rate on the training sample set (where sample weights are applied for weighting)

    Calculate the weight of a weak classifier based on its error rate (the higher the error rate, the lower the weight)

    Update all sample weights

At the end of the loop, we have weak classifiers (in this case, decision tree classifiers) as the number of "n_estimators". By weighting and adding these decision tree classifiers with weak classifier weights, we ultimately obtain an integrated strong classifier.

    Random Forest Classifier

An ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification
Experiment Results

Support Vector Machine (RBF Kernel)

To see if the SVM (RBF kernel) had a different success rate in classification depending on the time era of songs, the data was binned into different decades. An individual model was then trained on each bucket or era of data.

The data was scaled and standardized. The data was split 70% for a training set, and the other 30% was used as a test set, to score the accuracy of the model. The data sets were shuffled before splitting. The results of the models trained are in figure E.
Decade Bin 	Score
1960-1969 	0.7639799460084844
--- 	---
1970-1979 	0.7673819742489271
1980-1989 	0.7863000482392668
1990-1999 	0.8466183574879227
2000-2009 	0.8382519863791147
2010-2019 	0.8109375
All data in bins combined 	0.785598443074927

Figure E. SMV Scores on test set.

Out of all the data bins, 1990-1999 was most accurate. Overall, the model scored 0.78 when trained and tested for all data.

AdaBoost

The results of testing the model generated by the Adaboost classifier are as follows.
Decade Bin 	Score
1960-1969 	0.783262630158118
--- 	---
1970-1979 	0.7648068669527897
1980-1989 	0.7901591895803184
1990-1999 	0.8393719806763285
2000-2009 	0.843927355278093
2010-2019 	0.8296875
All data in bins combined 	0.8053843658773921

In addition, the accuracy of the test results of the model generated by Adaboost using Sklearn is 0.8023029516704508, which is not significantly different from the results of the model implemented using our code.
Comparison and Discussion

To assess the performance of the models, a few primary metrics are considered:

    ROC Curve/AUC: The ROC is a visual representation that displays how well a classification model works across different decision thresholds. It is a chart that shows two things: how often the model correctly identifies positive cases and how often it mistakenly identifies negative cases. [6].
    Accuracy: This metric provides a straightforward measure of the fraction of predictions our model got right.
    Recall: Recall represents the proportion of actual positive samples predicted to be positive samples.
    Precision: Precision refers to the proportion of predicted positive samples that are actually positive samples.
    F1_Score: Calculated from the precision and recall of the classification model, it is a metric that takes into account both of the above factors simultaneously.
    Confusion Matrix : Offers a detailed view of the true positives, true negatives, false positives, and false negatives. This metric is crucial as it provides insights into the type and frequency of errors made by the classifier.
    Calibration Curve: Calibration curves can assist in determining how accurate a binary classifier's model is, when trying to predict the class label. It is to determine if the model's confidence levels match up with how often it's correct. The X axis measures the "Mean predicted probability". Which is the value of the average predicted probability in that specific bin. The Y axis measures the "fractions of positives" indicating the percentage of how many songs were classified as a success in that bin. [7]
    Feature Importance: It assigns the score of input features based on their importance to predict the output. More the features will be responsible to predict the output more will be their score. [8]

Support Vector Machine (RBF Kernel)

Receiver operating characteristic (ROC) curve depicted in Figure F.

Figure F. SMV Model ROC Curve.

In Figure F, it can be examined that the support vector machine model trained, had an area under curve (AUC) of 0.87. For reference, a model of perfect accuracy will have an area of 1.0. (This value is not to be confused with the scores in Figure E).

Confusion Matrix of Error Rates/Accuracy.

Figure G. SMV Model Confusion Matrix.

A confusion matrix can determine and compare the amount of false positives, true positives, false negatives, and true negatives. From this matrix, we can ascertain the following rates in Figure H.

| | Formula | Rate | | --- | --- | --- |
True Positive 	TPR = TP / (TP + FN) 	0.73
True Negative 	TNR = TN / (TN + FP) 	0.85
False Negative 	FNR = FN / (TP + FN) 	0.15
False Positive 	FPR = FP / (FP + TN) 	0.27

Figure H. Precision rates.

Calibration Curve

Figure I. SMV Model Calibration Curve.

In figure I, it can be observed that around the mean predicted probability of ~0.25, the fraction of positives was too high. This implies that when the average predicted probability for that bin should have been around ~25%, the model classified around ~30% of songs as a success. Thus, it implies that the model is more prone to false positives around the predicted probability of that bin. Furthermore, at around the mean predicted probability of ~0.68, the fraction of positives was around ~0.58. Implying that the SMV model had accrued many false negative rates in that mean predicted probability bin. At around the mean predicted probability of 0.42, meaning the model is perfectly calibrated.

_ AdaBoost _

Receiver operating characteristic (ROC) curve

Figure J. Adaboost Model ROC Curve.

The AUC of Adaboost model generated by our own code is 0.8734487521143143.

Confusion Matrix of Error Rates/Accuracy/Recall/Precision.

Following this, a confusion matrix, precision, recall, and F1-score metrics for Adaboost model generated by our own code.

|
Matrix

|
Score

| | --- | --- | |
Accuracy

|
80.5%

| |
Recall

|
85%

| |
Precision

|
77.9%

| |
F1_score

|
81%

|

Figure O. Random Forest Model metrics.

Figure K. Adaboost Model Confusion Matrix.

| | Formula | Rate | | --- | --- | --- |
True Positive 	TPR = TP / (TP + FN) 	0.85
True Negative 	TNR = TN / (TN + FP) 	0.76
False Negative 	FNR = FN / (TP + FN) 	0.24
False Positive 	FPR = FP / (FP + TN) 	0.15

Figure L. Precision rates(Adaboost)..

_ Random Forest Classifier Analysis _

The dataset was partitioned as follows: 70% for training, 15% for validation, and 15% for testing. This split was achieved through randomized shuffling. Subsequently, the data was standardized to maintain consistency with the other two models in the analysis.

Confusion Matrix of Error Rates/Accuracy/Recall/Precision.

The initial model reported a training accuracy of 99% and a validation accuracy of 79%. Such a significant disparity between training and validation accuracies suggests potential overfitting. For further insights, a confusion matrix was generated based on the validation set.
Set 	Score
Training 	99%
Validation 	79%

Figure M. Random Forest Model accuracy score.

Figure N. Random Forest Model Confusion Matrix.

To optimize model performance, hyperparameter tuning was conducted using RandomSearchCV. The optimal parameters yielded a validation accuracy of 79%. When these parameters were applied to the testing set, the model achieved a testing accuracy of 79% and a revised training accuracy of 94%, indicating a slight reduction in training accuracy.
Set 	Score
Training 	94%
Validation 	79%
Test 	79%

Figure O. Random Forest Model accuracy score.

Following this, a confusion matrix, precision, recall, and F1-score metrics were calculated on the test set using the best parameters.



Figure O. Random Forest Model other metrics.

Feature Importance/ Receiver operating characteristic (ROC) curve

In the final analysis phase, both feature importance and the Receiver Operating Characteristic (ROC) curve were plotted. This visualization provides a clear representation of the significance of each feature during the model training process.

Figure P. Random Forest Model Feature Importance.

Figure Q. Random Forest Model ROC Curve.
Conclusion

As even the worst machine learning model, Support Vector Machine (RBF kernel), had an accuracy of 78%, it can still be alluded to the fact that there is a very strong correlation between the acoustic attributes of a song, and whether the song will be a success or not. This not only informs us of a common spirit, or a hidden formulaic thought process among people, but may also tell us secrets in how to produce commercial success from music. In the future, we can use supervised learning algorithms to determine precise values of individual acoustic attributes to create a successful song.

The model generated using the Adaboost algorithm has a higher prediction accuracy than SVM,
Considering the outcomes, the Random Forest model's overall accuracy of 0.7974 and F1-score of 0.80 indicate that it is functioning quite well. The model's precision of 0.7749 and recall of 0.8538 also show that it can detect a large percentage of positive events while minimizing false positives.
Future Enhancements
1. More data could help the model perform better in terms of accuracy and generalization.
2. To enhance the model's capacity to identify patterns in the data, consider investigating new features or modifying existing ones.
Sources

[1] https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-py

[2] https://stats.stackexchange.com/questions/46855/where-does-the-definition-of-the-hyperplane-in-a-simple-svm-come-from

[3] "A primer on kernel methods".Kernel Methods in Computational Biology, pg. 23

[4] STA561: Probabilistic machine learning: Kernels and Kernel Methods (10/09/13)

[5] https://www.pycodemates.com/2022/10/the-rbf-kernel-in-svm-complete-guide.html

[6] https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc

[7] https://scikit-learn.org/stable/modules/calibration.html#calibration

[8] https://medium.com/analytics-vidhya/feature-importance-explained-bfc8d874bcf

[9]https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=A%20random%20forest%20is%20a,accuracy%20and%20control%20over%2Dfitting.

[10]https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

[11]https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
