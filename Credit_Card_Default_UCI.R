# credit card default

library("lattice")
library("ggplot2")
library("caret") #for confusion matrix
library(ROSE) #required for balancing imbalanced data
library(randomForest) #library for Random FOrest
library(e1071) #library for SVM Model
library(nnet) #library for neural networks
library(pROC) #for area under curve required to test accuracy of results of an imbalanced data
library(xgboost)
library(dplyr)
library(ggplot2)
library(coefplot) #for plotting coefficients of glm model

#read data
mydata<-read.csv("UCI_Credit_Card.csv")
head(mydata)
nrow(mydata)

#check and remove if there is any missing data
sapply(mydata, function(x) sum(is.na(x)))
mydata<-na.omit(mydata)
nrow(mydata)

#create formula
formula_cc<-default.payment.next.month~ID+LIMIT_BAL+SEX+EDUCATION+MARRIAGE+AGE+PAY_0+PAY_2+PAY_3+PAY_4+PAY_5+PAY_6+BILL_AMT1+BILL_AMT2+BILL_AMT3+BILL_AMT4+BILL_AMT5+BILL_AMT6+PAY_AMT1+PAY_AMT2+PAY_AMT3+PAY_AMT4+PAY_AMT5+PAY_AMT6

#Check sample skewness
table(mydata$default.payment.next.month)

#Split data in train and test
set.seed(1234)
ind<-sample(2,nrow(mydata),replace=TRUE,prob = c(0.7,0.3))
train_data<-mydata[ind==1,]
test_data<-mydata[ind==2,]
table(train_data$default.payment.next.month)

#data balancing (not required as the accuracy went down)
#train_data<- ovun.sample(formula_cc, data = train_data, method = "under", N = 10000, seed = 1)$data
#table(train_data$default.payment.next.month)

#Logistic regression
glm_cc<-glm(formula_cc,binomial(link = 'logit'),data = train_data)
glm_cc
predicted_glm<-predict(glm_cc,test_data,type = "response")
head(predicted_glm)
for (i in 1:length(predicted_glm)){if(predicted_glm[i]<0.5){predicted_glm[i]=0} else{predicted_glm[i]=1}}
predictability_glm<-sum(predicted_glm==test_data$default.payment.next.month)/length(test_data$default.payment.next.month)
predictability_glm
coefplot(glm_cc)

#Importance of factors
ggplot(glm_cc$coefficients, aes(x=reorder(glm_cc$coefficients), y=glm_cc$terms))+geom_bar(stat='identity')+coord_flip()

#random Forest
#model_rf<- randomForest(formula_cc,data=train_data,importance=TRUE,ntree=200)
#pred_rf<- predict(model_rf_under, test_data)
#for (i in 1:length(pred_rf)){if(pred_rf[i]<0.5){pred_rf[i]=0} else{pred_rf[i]=1}}
#predictability_rf<-sum(pred_rf==test_data$default.payment.next.month)/length(test_data$default.payment.next.month)
#predictability_rf

#Support Vector Machines
model_svm<- svm(formula_cc,data=train_data)
pred_svm<- predict(model_svm, test_data)
for (i in 1:length(pred_svm)){if(pred_svm[i]<0.5){pred_svm[i]=0} else{pred_svm[i]=1}}
predictability_svm<-sum(pred_svm==test_data$default.payment.next.month)/length(test_data$default.payment.next.month)
predictability_svm


#Xgboost
train_variables<-as.matrix(train_data[,-25])
train_label<-train_data[,25]
train_matrix<-xgb.DMatrix(data=train_variables,label=train_label)

test_variables<-as.matrix(test_data[,-25])
test_label<-test_data[,25]
test_matrix<-xgb.DMatrix(data=test_variables,label=test_label)

numberOfClasses <- length(unique(mydata$default.payment.next.month))
xgb_params <- list("objective" = "multi:softmax","eval_metric" = "mlogloss","num_class" =numberOfClasses,eta=0.3,max_depth=6)
bst<-xgboost(params = xgb_params,data=train_variables,label = train_label,nrounds = 50,verbose = 0)
test_predict_xgb<-predict(bst,newdata = test_matrix)
for (i in 1:length(test_predict_xgb)){if(test_predict_xgb[i]<0.5){test_predict_xgb[i]=0} else{test_predict_xgb[i]=1}}
predictability_xgboost<-(sum(test_predict_xgb==test_data$default.payment.next.month))/length(test_data$default.payment.next.month)
predictability_xgboost

#Importance of Factors
names <-  colnames(mydata[,-1])
importance_matrix = xgb.importance(feature_names = names, model = bst)
gp_xgb<-xgb.plot.importance(importance_matrix)



#Random Forrest MOdel
model_rf<- randomForest(formula_cc,data=train_data,importance=TRUE,ntree=200)
pred_rf<- predict(model_rf, test_data)
for (i in 1:length(pred_rf)){if(pred_rf[i]<0.5){pred_rf[i]=0} else{pred_rf[i]=1}}
predictability_rf<-sum(pred_rf==test_data$default.payment.next.month)/length(test_data$default.payment.next.month)
predictability_rf
