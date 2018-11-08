# Read the data from the CSV file
Customer_Org1<- read.csv('C:/Users/HARI/Desktop/Neelima/Reva/Phani/Project/term3.csv', na.strings=c("","NA"))
Customer_test<- read.csv('C:/Users/HARI/Desktop/Neelima/Reva/Phani/Project/Term end Project.csv', na.strings=c("","NA"))

# No misssing values ; clean data so No imputation required
Customer_Org<- Customer_Org1
colSums(is.na(Customer_Org))

# Preliminary Analysis 
str(Customer_Org)
summary(Customer_Org)
cor(Customer_Org_Num,Customer_Org$Reached.on.Time_Y.N)

# Covert the no of calls to factors 
Customer_Org$Customer_care_calls_fac<-factor(Customer_Org$Customer_care_calls)
Customer_Org$Customer_rating_fac <- factor(Customer_Org$Customer_rating)
Customer_Org$Prior_purchases_fac <- factor(Customer_Org$Prior_purchases)
Customer_Org$Reached.on.Time_Y.N_fac<-factor(Customer_Org$Reached.on.Time_Y.N)

# Covert the no of calls to factors 
Customer_test$Customer_care_calls_fac<-factor(Customer_test$Customer_care_calls)
Customer_test$Customer_rating_fac <- factor(Customer_test$Customer_rating)
Customer_test$Reached.on.Time_Y.N_fac<-factor(Customer_test$Reached.on.Time_Y.N)

Customer_Org_Num <- Customer_Org[c("Cost_of_the_Product","Discount_offered", "Weight_in_gms")]
Customer_Org_Num_Scaled <- Customer_Org_Num %>% scale()
Customer_Org_t <- cbind(Customer_Org, Customer_Org_Num_Scaled)

#Plot the basic graphs 

plot(Customer_Org$Warehouse_block,Customer_Org$Reached.on.Time_Y.N)
plot(Customer_Org$Mode_of_Shipment)
plot(Customer_Org$Customer_care_calls_fac)
plot(Customer_Org$Customer_rating_fac)
plot(Customer_Org$Cost_of_the_Product,Customer_Org$Reached.on.Time_Y.N)
str(Customer_Org)

Coloums <- names(Customer_Org_t)
predictors<-Coloums[c(2,3,8,9,13,14,15)]
predictors1<-Coloums[c(2,3,8,9,13,14,15,17,18,19)]
outcomeName <-Coloums[16]
outcomeName
Coloums
predictors1
names(Customer_Org_t)

#Include Caret library
library(caret)
library(dplyr)
library(ggplot2)
library(lattice)

#Spliting training set into two parts based on outcome: 75% and 25%
index <- createDataPartition(Customer_Org$Reached.on.Time_Y.N, p=0.75, list=FALSE)
trainSet <- Customer_Org[ index,]
testSet <- Customer_Org[-index,]
head(trainSet)
table(trainSet$Reached.on.Time_Y.N)

#Spliting training set into two parts based on outcome: 75% and 25%
index <- createDataPartition(Customer_Org_t$Reached.on.Time_Y.N, p=0.75, list=FALSE)
trainSet <- Customer_Org_t[ index,]
testSet <- Customer_Org_t[-index,]
names(trainSet)
table(trainSet$Reached.on.Time_Y.N)
names(predictors1)
head(testSet)
#Build the Models:

model_glm<-train(trainSet[,predictors],trainSet[,outcomeName],method='glm')
model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf')

model_glm1<-train(trainSet[,predictors1],trainSet[,outcomeName],method='glm')

summary(model_glm)
summary(model_rf)
anova(model_glm1)

varImp(object=model_glm)
varImp(object=model_rf)


plot(varImp(object=model_glm),main="LOGISTIC - Variable Importance")
plot(varImp(object=model_rf),main="RF - Variable Importance")

#get the test
glm_predict <- predict(model_glm,newdata = testSet[,predictors],type = "raw")
pr_glm <- prediction(as.numeric(glm_predict),as.numeric(testSet$Reached.on.Time_Y.N_fac))

glm1_predict <- predict(model_glm1,newdata = testSet[,predictors1],type = "raw")
pr_glm1 <- prediction(as.numeric(glm1_predict),as.numeric(testSet$Reached.on.Time_Y.N_fac))


perf_glm1 <- performance(pr_glm1,measure = "tpr",x.measure = "fpr") 
plot(perf_glm1) 
auc(as.numeric(testSet$Reached.on.Time_Y.N_fac),as.numeric(glm1_predict))

##plot ROC 
library(ROCR)
install.packages("Metrics")
library(Metrics)

perf_glm <- performance(pr_glm,measure = "tpr",x.measure = "fpr") 
plot(perf_glm) 
auc(as.numeric(testSet$Reached.on.Time_Y.N_fac),as.numeric(glm_predict))

#get the test for random forest

rf_predict <- predict(model_rf,newdata = testSet[,predictors],type = "raw")
pr_rf <- prediction(as.numeric(rf_predict),as.numeric(testSet$Reached.on.Time_Y.N_fac))
perf_rf <- performance(pr_rf,measure = "tpr",x.measure = "fpr") 
plot(perf_rf) 
auc(as.numeric(testSet$Reached.on.Time_Y.N_fac),as.numeric(rf_predict))


#SVM

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
training <- trainSet[c(2,3,8,9,13,14,15,16)]
set.seed(3233)
head(training)
svm_Linear <- train(Reached.on.Time_Y.N_fac ~., data = training, method = "svmLinear",trControl=trctrl,preProcess = c("center", "scale"),tuneLength = 10)
summary(svm_Linear)


svm_predict <- predict(svm_Linear,newdata = testSet[,predictors],type = "raw")
pr_svm <- prediction(as.numeric(svm_predict),as.numeric(testSet$Reached.on.Time_Y.N_fac))
perf_svm <- performance(pr_svm,measure = "tpr",x.measure = "fpr") 
plot(perf_svm) 
auc(as.numeric(testSet$Reached.on.Time_Y.N_fac),as.numeric(svm_predict))

#Xgboost Implementation

library(xgboost)
install.packages("statsr")
library(statsr)

dtrain <- sparse.model.matrix(Reached.on.Time_Y.N_fac ~ .,lable = Reached.on.Time_Y.N_fac, 
                              data = trainSet(, predictors))

dtrain <- sparse.model.matrix(Reached.on.Time_Y.N_fac ~ .,lable = Reached.on.Time_Y.N_fac, data = trainSet(, ))
dtest <- sparse.model.matrix(Reached.on.Time_Y.N_fac ~ .,lable = Reached.on.Time_Y.N_fac, data = testSet)
trainlable = trainSet[,"Reached.on.Time_Y.N"]
testlable = testSet[,"Reached.on.Time_Y.N"]
                      
ddtrain <- xgb.DMatrix(data = dtrain, label=trainlable)
ddtest <- xgb.DMatrix(data = dtest, label=testlable)
watchlist <- list(train=ddtrain, test=ddtest)

bst <- xgb.train(data=ddtrain, max_depth=4, eta=1, nthread = 2, nrounds=2, watchlist=watchlist, objective = "binary:logistic")

bst_predict <- predict(bst,newdata = ddtest)
pr_bst <- prediction(as.numeric(rf_predict),as.numeric(testSet$Reached.on.Time_Y.N_fac))
perf_bst <- performance(pr_bst,measure = "tpr",x.measure = "fpr") 
plot(perf_bst) 
auc(as.numeric(testSet$Reached.on.Time_Y.N_fac),as.numeric(bst_predict))

#Problem Statement 2

#Good Customer Rating:
plot(Customer_Org$Customer_rating_fac,Customer_Org$Reached.on.Time_Y.N_fac, xlab= 'Customer Ratings', ylab ='Product Delivered')
plot(Customer_Org$Reached.on.Time_Y.N_fac,Customer_Org$Cost_of_the_Product)
plot(Customer_Org$Customer_care_calls_fac,Customer_Org$Reached.on.Time_Y.N_fac)
plot(Customer_Org$Reached.on.Time_Y.N_fac,Customer_Org$Customer_care_calls_fac)
plot(Customer_Org$Prior_purchases_fac,Customer_Org$Reached.on.Time_Y.N_fac)
plot(Customer_Org$Reached.on.Time_Y.N_fac,Customer_Org$Cost_of_the_Product)


#Problem Statement 3
install.packages("factoextra")
install.packages("cluster")
install.packages("magrittr")

library("cluster")
library("factoextra")
library("magrittr")
library(dplyr)


delay_Customer <- Customer_Org1[Customer_Org1$Reached.on.Time_Y.N == 0,]
str(delay_Customer)
Num_Customer_select <- delay_Customer[c(4,5,6,7,10,11)]
Scaled_Customer_select <- Num_Customer_select  %>% scale()
str(Scaled_Customer_select)
str(Num_Customer_select)
res.dist <- get_dist(Scaled_Customer_select, stand = TRUE, method = "pearson")
fviz_dist(res.dist,gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

#Determining the optimal number of clusters: use factoextra::fviz_nbclust()
fviz_nbclust(Num_Customer_select, kmeans, method = "gap_stat")

set.seed(123)
km.res <- kmeans(Num_Customer_select, 3, nstart = 25)
# Visualize
library("factoextra")
fviz_cluster(km.res, data = Num_Customer_select,ellipse.type = "convex",palette = "jco",ggtheme = theme_minimal())
Num_Customer_select$cluster <- km.res$cluster
boxplot(Num_Customer_select$cluster)

str(km.res)
km.res$cluster
km.res$size
Num_Customer_select$cluster<- km.res$cluster
boxplot(Num_Customer_select$Customer_rating~as.factor(Num_Customer_select$cluster), data = Num_Customer_select)
boxplot(Num_Customer_select$Customer_care_calls~as.factor(Num_Customer_select$cluster), data = Num_Customer_select)
#boxplot(Num_Customer_select$Cost_of_the_Product~as.factor(Num_Customer_select$cluster), data = Num_Customer_select)
boxplot(Num_Customer_select$Prior_purchases~as.factor(Num_Customer_select$cluster), data = Num_Customer_select)
#boxplot(Num_Customer_select$Discount_offered~as.factor(Num_Customer_select$cluster), data = Num_Customer_select)
boxplot(Num_Customer_select$Weight_in_gms~as.factor(Num_Customer_select$cluster), data = Num_Customer_select)


table(Num_Customer_select$cluster)
# Dissimilarity matrix
#pHANI'S CODE

d <- dist(Num_Customer_select, method = "euclidean")

# Hierarchical clustering using Complete Linkage
hc1 <- hclust(d, method = "complete" )
hc1
hc1$labels

# Plot the obtained dendrogram
plot(hc1, cex = 0.6, hang = -1)

#Compute with agnes
hc2 <- agnes(Num_Customer_select, method = "complete")
hc2 <- agnes(Num_Customer_select, method = "ward")
hc2$ac

nstall.packages("tidyverse")
install.packages("cluster")
install.packages("factoextra")
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering visualization
hc2
# Agglomerative coefficient
hc2$ac
# methods to assess
m <- c( "average", "single", "complete", "ward")
m
names(m) <- c( "average", "single", "complete", "ward")

# function to compute coefficient
ac <- function(x) {
  agnes(Num_Customer_select,method = x)$ac
}
ac

#ac1 <- map_dbl(m, ac)

ac1<- ac(m)
ac1

hc3 <- agnes(Num_Customer_select, method = "ward")
pltree(hc3, cex = 0.6, hang = -1, main = "Dendrogram of agnes")
pltree(hc3, cex = 0.6, hang = -1, main = "Dendrogram of agnes")

plot(hc2, cex = 0.6)
pltree(hc2, cex = 0.6, hang = -1, main = "Dendrogram of diana")

hc5 <- hclust(d, method = "ward.D2" )
fviz_nbclust(Num_Customer_select, FUN = hcut, method = "wss")
sub_grp <- cutree(hc5, k = 3)
table(sub_grp)
Num_Customer_select$cluster <-sub_grp
str(Num_Customer_select)
rect.hclust(hc5, k = 4, border = 2:5)
plot(hc5,cex=0.6)
#Rough Work
write.csv(Num_Customer_select,"'C:/Users/HARI/Desktop/Neelima/Reva/Phani/Project/new.csv")

trctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
head(trainlable)
head(trainSet[,"Reached.on.Time_Y.N_fac"])

sparse_matrix <- sparse.model.matrix(Reached.on.Time_Y.N_fac ~ Warehouse_block+Mode_of_Shipment+Cost_of_the_Product+Prior_purchases+Product_importance+Gender+Discount_offered+Weight_in_gms+Customer_care_calls_fac+Customer_rating_fac, data = training)
y_lable= training[c(11)]
head(y_lable)
str(sparse_matrix)

xgb <- xgboost(data = sparse_matrix, 
               label = y_lable, 
               eta = 0.1,
               max_depth = 15, 
               nround=25, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "merror",
               objective = "multi:softprob",
               num_class = 12,
               nthread = 3
)




library(e1071)
model_svm <- svm( Reached.on.Time_Y.N_fac ~ Warehouse_block+Mode_of_Shipment+Cost_of_the_Product+Prior_purchases+Product_importance+Gender+Discount_offered+Weight_in_gms+Customer_care_calls_fac+Customer_rating_fac, trainSet)
model_xgb <- xgboost(data = data.matrix(X[,-1]), label = y, eta = 0.1,max_depth = 15,nround=25,subsample = 0.5,
               colsample_bytree = 0.5, seed = 1,eval_metric = "merror",objective = "multi:softprob",num_class = 12,
               nthread = 3)
lm(Customer_Org1$Reached.on.Time_Y.N~Customer_Org[-1])

logisticresults <- glm(Reached.on.Time_Y.N ~ Warehouse_block+Mode_of_Shipment+Cost_of_the_Product+Prior_purchases+Product_importance+
                         Gender+Discount_offered+Weight_in_gms+Customer_care_calls+Customer_rating,
                       data = Customer_Org1, family=binomial(link=logit))
anova(logisticresults)

fitted.results <- predict(model,newdata=subset(testSet,select=c(2,3,4,5,6,7,8)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != test$Survived)
print(paste('Accuracy',1-misClasificError))


summary(model_svm)
model_svm
anova(model_svm,test='Chisq')

varImp(object=model_svm)
varImp(object=model_xgb)



#model_svm<-train(trainSet[,predictors],trainSet[,outcomeName],method='svmLinear', trControl=trctrl,
#                 preProcess = c("center", "scale"), tuneLength = 2)
model_xgb<-train(trainSet[,predictors],trainSet[,outcomeName],method='xgbTree')

#fitControl <- trainControl( method = "repeatedcv",  number = 5,  repeats = 5)
#model_gbm1<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=10)
#varImp(object=model_gbm1)
#plot(varImp(object=model_gbm1),main="GBM - Variable Importance")


#test-data
glmtest<-predict.train(object=model_glm,testSet[,predictors],type="raw")
rfrtest<-predict.train(object=model_rf,testSet[,predictors],type="raw")
xgbtest<-predict.train(object=model_xgb,testSet[,predictors],type="raw")

Scaled_Customer_select <- delay_Customer_select$Discount_offered  %>% scale()


