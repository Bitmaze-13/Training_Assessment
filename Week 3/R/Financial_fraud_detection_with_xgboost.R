# Load required libraries for data manipulation and visualization
library(data.table)  # Efficient data manipulation
library(ggplot2)     # Data visualization
library(psych)       # Psychological, psychometric, and personality research
library(GGally)      # Extensions to ggplot2
library(dplyr)       # Data manipulation
library(cowplot)     # Plot arrangement
library(caret)       # Classification and regression training
library(pROC)        # ROC curve and AUC calculations
library(ROCR)        # Visualizing the performance of scoring classifiers
library(MASS)        # Functions and datasets for Venables and Ripley's MASS book
#library(dummies)     # Create dummy/indicator variables
library(class)       # K-nearest neighbors classification
library(xgboost)     # Extreme Gradient Boosting
library(e1071)       # Misc functions from the Department of Statistics, up
library(nnet)        # Feed-forward neural networks and multinomial log-linear models


credit=fread("Dataset.csv")
head(credit)

credit=credit[,-1]#There's no use of the column id in our analysis


sum(is.na(credit))

names(credit)
names(credit)[6]="PAY_1"
names(credit)[24] = "target"


df=as.data.frame(credit)
df[c("SEX","MARRIAGE","EDUCATION","target","PAY_1","PAY_2","PAY_3","PAY_4","PAY_5",
     "PAY_6")]= 
  lapply(df[c("SEX","MARRIAGE","EDUCATION","target","PAY_1","PAY_2","PAY_3","PAY_4",
              "PAY_5","PAY_6")]
         ,function(x) as.factor(x))                                                                       



credit=df
rm(df)


df=as.data.frame(data.matrix(credit[,c(-2:-4,-6:-11,-24)]))
ggcorr(df,method=c("everything", "pearson"))+ggtitle("Correlation Steps")

rm(df)



ggplot(credit,aes(x=credit$LIMIT_BAL,fill=credit$target))+
  geom_density(alpha=0.6,show.legend = T,color="blue")+
  ggtitle("Density plot oh Credit Amount")+
  xlab("Credit Amount")


ggplot(credit,aes(x=credit$AGE,fill=credit$target))+
  geom_histogram(show.legend = T,alpha=0.9)+
  ggtitle("AGE for different customers with respect to default")+
  xlab("AGE")

ggplot(credit,aes(x=credit$MARRIAGE,group=credit$target))+
  geom_bar(show.legend = T,fill="lightblue")+
  ggtitle("Default for different marital status")+
  xlab("Marriage")+
  facet_grid(~credit$target)

q=list()             #creating empty plot list
for(i in 6:11){
  q[[i]]=ggplot(credit,aes(x=credit[,i],y=credit[,1],
                           color=credit$target,palette="jco"))+
    geom_point(show.legend = T)+
    xlab(paste0("PAY_",i-5,sep=""))+
    ylab("Limit Bal")+
    ggtitle(paste0("PAY_",i-5,"Vs Limit Bal",sep=""))
}

plot_grid(q[[6]],q[[7]],q[[8]],q[[9]],q[[10]],q[[11]],nrow=3,ncol=2)

credit$EDUCATION = recode_factor(credit$EDUCATION, '4' = "0", '5' = "0", '6' = "0",
                                 .default = levels(credit$EDUCATION))
credit$MARRIAGE = recode_factor(credit$MARRIAGE, '0'="3",
                                .default = levels(credit$MARRIAGE))
#Partitioning the whole data in quantitative and qualitative parts defining the target
quanti=credit[,c(-2:-4,-6:-11,-24)]
quali=credit[,c(2:4,6:11)]
target=credit$target
(table(target)/length(target))

all.features=cbind(quanti,quali,target)
head(all.features)


set.seed(666)#for reproducability of result
ind=sample(nrow(all.features),24000,replace = F)

train.logit=all.features[ind,]
test.logit=all.features[-ind,]

model.logit=glm(target~.,data=train.logit,family="binomial")

summary(model.logit)

pred.logit=predict(model.logit,type="response",newdata = test.logit)

pred.def=ifelse(pred.logit>0.5,"1","0")
pred.def=ifelse(predict(model.logit,type="response",newdata = train.logit)>0.5,"1","0")


par(mfrow=c(1,2))
par(pty="s")

# For training
roc(train.logit$target,model.logit$fitted.values,plot=T,col="#69b3a2",
    print.auc=T,legacy.axes=TRUE,percent = T,
    xlab="False Positive percentage",ylab="True Positive percentage",
    lwd=5,main="Train Set")

# for testing

roc(test.logit$target,pred.logit,plot=T,col="navyblue",print.auc=T,legacy.axes=TRUE,
    percent = T,
    xlab="False Positive percentage",ylab="True Positive percentage",
    lwd=5,main="Test Set")