#--------------------#Prediction using KNN-----------------------------------

d=read.csv("~/Downloads/diabetes.csv")
View(d)
dim(d)
sum(is.na(d)) 

# checking for percent of having diabetics
sum(d$Outcome)/dim(d)[1]
table(d$Outcome)
500/(500+268)  #0.6510417 checking for percent of not having diabetics

outcome = as.data.frame( d[ , 9] )
colnames(outcome) = c("Outcome")
View(outcome)
class(outcome$Outcome)    
outcome$Outcome <- factor(outcome$Outcome)
class(outcome$Outcome)

d <- d[ , -9]
View(d)
View(outcome)
library(caret)
norm.values <- preProcess(d, method=c("center", "scale"))
d.norm <- predict(norm.values, d)  
View(d.norm)  


set.seed(2023)
total.rows = dim(d.norm)[1]
train.rows = sample( 1:total.rows, total.rows*0.6 )
valid.rows = setdiff(1:total.rows, train.rows)   # setdiff() function to subtract one set from the other
train.df = d.norm[train.rows, ]
valid.df = d.norm[-train.rows, ]

library(FNN)
mydata <- data.frame(k = seq(1, 10), accuracy = rep(0, 10), sensitivity = rep(0, 10), specificity = rep(0, 10))
View(mydata)
myk = 1:10    

for(i in myk) {     
  
  myknn <- knn(train.df, test = valid.df, 
               cl = outcome[train.rows,], k = i)                    
  
  a = confusionMatrix(myknn, outcome[valid.rows,], positive = "1")    
 
  mydata[i, 1] = i
  mydata[i, 2] = a$overall[1]  # Accuracy
  mydata[i, 3] = a$byClass[1]  # Sensitivity
  mydata[i, 4] = a$byClass[2]  # Specificity
  
}
View(mydata)

knn.pred <- knn(train = train.df, test = valid.df, 
                cl = outcome[train.rows,], k = 5)     
class(knn.pred)     
p=data.frame(Actual=outcome[valid.rows,], Predicted = knn.pred)
View(p)

table(p)

a=confusionMatrix(knn.pred, outcome[valid.rows,], positive = "1")   # 1 means diabetics
a

a$table       # confusion matrix
a$overall[1]  
# Accuracy 0.7305195
a$byClass[1] 
#Sensitivity 0.5462963 
a$byClass[2]  
#Specificity 0.83 



#------------------------------Prediction using Naive Bayes-----------------------------
d <- read.csv("~/Downloads/diabetes.csv", header = TRUE)    
View(d)

d$Outcome <- ifelse(d$Outcome == 0, "No", "Yes")
d$Outcome <- factor(d$Outcome)
d$Pregnancies <- factor(d$Pregnancies)

set.seed(123)
train.index <- sample(c(1:dim(d)[1]), dim(d)[1]*0.6)  
train.df <- d[train.index, ]
valid.df <- d[-train.index, ]

library(e1071)

d.nb = naiveBayes(Outcome ~ . , data = train.df)
d.nb

table(train.df$Outcome)/dim(train.df)[1]
pred.prob = predict(d.nb, newdata = valid.df, type = "raw")    
pred.class = predict(d.nb, newdata = valid.df, type = "class") 
results = data.frame(actual = valid.df$Outcome, predicted=pred.class, probability = pred.prob )
library(caret)
pred.class = predict(d.nb, newdata = valid.df, type = "class")  # Calculate predicted classes in the validation
confusionMatrix(pred.class, valid.df$Outcome, positive = "Yes")
#Accuracy : 0.7597      
#Sensitivity : 0.6078         
#Specificity : 0.8350  


library(caret)

pred.prob = predict(d.nb, newdata = valid.df, type = "raw")   # get propensities
pred.prob = data.frame(pred.prob)

mylift <- lift(relevel(valid.df$Outcome, ref="Yes") ~ pred.prob$Yes)
xyplot(mylift, plot = "gain")

# ROC
library(pROC)

results = data.frame(actual = valid.df$Outcome, predicted=pred.class, probability = pred.prob )

myroc <- roc(results$actual , results$probability.Yes, levels = c(Yes,No))
myroc <- roc(results$actual , results$probability.Yes, levels = c("Yes","No"))

plot(myroc, main = "ROC curve for the model",
     col = "blue", lwd = 2, legacy.axes = TRUE)    # if add = TRUE, you can add more charts on top of each

auc(myroc)

#-----------------------------------------------------------------------------
#Purpose/Business Objective/Problem:

#The business objective is to create a predictive tool that assists medical professionals in identifying individuals at 
#risk of diabetes early, enabling timely intervention and preventive measures. 
#Early detection and management of diabetes can significantly improve patients' quality of life and reduce the risk of complications associated with the disease.
#--------------------------------------------------------------------------------
#the Naïve Bayes model performed better compared to the K-Nearest Neighbors (KNN) model in terms of accuracy, sensitivity, and specificity. 
#From the accuracy standpoint, the Naïve Bayes model achieved a higher accuracy score (0.7597) compared to the KNN model (0.7305). 
#Additionally, the Naïve Bayes model demonstrated better sensitivity (0.6078) and specificity (0.8350) compared to the KNN model (sensitivity: 0.5463, specificity: 0.83).
#-------------------------------------------------------------------------------
#Difficulties arose when making trade-offs between model complexity and performance, as well as managing the interplay of data preprocessing, algorithm selection, and parameter tuning. Nevertheless, these challenges offered valuable insights into the complexities of real-world data analysis, 
#underscoring the significance of iterative refinement and a strong understanding of the underlying concepts.
#This assignment provided practical exposure to machine learning algorithms,I gained valuable experiences and insights into their performance and challenges. The primary objective was to assess the accuracy and specificities of these models in predicting outcomes.


