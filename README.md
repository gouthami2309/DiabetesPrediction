Diabetes Prediction Models
This project focuses on predicting diabetes using two machine learning algorithms: K-Nearest Neighbors (KNN) and Naive Bayes. The aim is to assist medical professionals in identifying individuals at risk of diabetes early, enabling timely intervention and preventive measures.

Table of Contents
Introduction
Datasets
Requirements
Installation
Usage
Models and Performance
Challenges

Introduction
The primary objective of this project is to create a predictive tool that can help in the early detection of diabetes. By leveraging machine learning algorithms, we can predict the likelihood of a patient having diabetes based on various medical attributes.

Datasets
diabetes.csv: This dataset contains various medical attributes of patients along with the target variable indicating whether the patient has diabetes or not.
Requirements
R (version 4.0.0 or higher)
The following R packages:
caret
dplyr
e1071
FNN
pROC
ggplot2
Installation
Install R from CRAN.
Install the required packages by running the following commands in R:
R
Copy code
install.packages("caret")
install.packages("dplyr")
install.packages("e1071")
install.packages("FNN")
install.packages("pROC")
install.packages("ggplot2")
Usage
Clone this repository to your local machine.
Place the diabetes.csv file in a suitable directory.
Update the file paths in the script to point to the correct location of your CSV file.
Run the script in R or RStudio.
Models and Performance
K-Nearest Neighbors (KNN)
The KNN model was implemented and evaluated with different values of k to determine the optimal parameter.

Performance:

Accuracy: 0.7305
Sensitivity: 0.5463
Specificity: 0.83
Naive Bayes
The Naive Bayes model was implemented and evaluated for comparison with the KNN model.

Performance:

Accuracy: 0.7597
Sensitivity: 0.6078
Specificity: 0.8350
Challenges
Some challenges faced during the project included:

Trade-offs between model complexity and performance.
Managing data preprocessing, algorithm selection, and parameter tuning.
Iterative refinement to improve model accuracy and robustness.
