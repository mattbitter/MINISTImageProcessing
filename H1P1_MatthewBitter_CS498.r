
#Install Pacakges
install.packages("e1071",repos = "http://cran.us.r-project.org")

#Load flat file
data.table <- read.csv(file="C:/Users/Splinta/GoogleDrive/Masters/Courses/CS498/Assignments/HW1 P1/Book1.csv",header=TRUE)

#Copy data and load libraries and seperate lables from features
wdat <- data.table

library(klaR)
library(caret)
library(e1071)

bigx<-wdat[,-c(9)]
bigy<-wdat[,9]
trscore<-array(dim=10)
tescore<-array(dim=10)

#Question 1 , Part A  #############################################################
#Source Reference : http://luthuli.cs.uiuc.edu/~daf/courses/AML-18/aml-home.html

for (wi in 1:10)
{wtd<-createDataPartition(y=bigy, p=.8, list=FALSE) #create list of indexes for 80% of data
 nbx<-bigx # copy all x's to new list
 
 ntrbx<-nbx[wtd, ] #apply index to x and create new list to filter x to 80% - train
 ntrby<-bigy[wtd] #apply index to y and create new list to filter y to 80% - train
 trposflag<-ntrby>0 # list of y's with TRUE/FLASE for people with diabeties - train
 ptregs<-ntrbx[trposflag, ] # filter x for people with diabetes - train
 ntregs<-ntrbx[!trposflag,] # filter x for people without diabetes- train
 
 #Calculate the P(y) and insert into ppy for positive diabetis and npy for negative diabetes
 #then add ppy and npy to the loglikelyhood based on norm distribution lower down in the code
 ppy <- log(nrow(ptregs)/nrow(ntrbx))
 npy <- log(nrow(ntregs)/nrow(ntrbx))
 
 ntebx<-nbx[-wtd, ] #create test for x
 nteby<-bigy[-wtd] #create test for y
 ptrmean<-sapply(ptregs, mean, na.rm=TRUE) #mean for all features for people with diabetes - trian
 ntrmean<-sapply(ntregs, mean, na.rm=TRUE) #mean for all features for people without diabetes - trian
 ptrsd<-sapply(ptregs, sd, na.rm=TRUE)  #sd for all features for people with diabeties - trian
 ntrsd<-sapply(ntregs, sd, na.rm=TRUE)#sd for all features for people without diabeties - trian
 ptroffsets<-t(t(ntrbx)-ptrmean) #subtract mean from all people, mean is from people with diabetes? 
 ptrscales<-t(t(ptroffsets)/ptrsd)# takes the features after mean subtracting and divides by sd only for diabetes - training
 ptrlogs<--( (1/2)*rowSums(apply(ptrscales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd)) ) + ppy # loglikelyhood 
 ntroffsets<-t(t(ntrbx)-ntrmean) #non dia
 ntrscales<-t(t(ntroffsets)/ntrsd) #non dia sd norm
 ntrlogs<--( (1/2)*rowSums(apply(ntrscales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd)) ) + npy# loglikelyhood 
 lvwtr<-ptrlogs>ntrlogs
 gotrighttr<-lvwtr==ntrby
 trscore[wi]<-sum(gotrighttr)/(sum(gotrighttr)+sum(!gotrighttr))
 pteoffsets<-t(t(ntebx)-ptrmean)
 ptescales<-t(t(pteoffsets)/ptrsd)
 ptelogs<--( (1/2)*rowSums(apply(ptescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd)) ) + ppy# loglikelyhood 
 nteoffsets<-t(t(ntebx)-ntrmean)
 ntescales<-t(t(nteoffsets)/ntrsd)
 ntelogs<--( (1/2)*rowSums(apply(ntescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd)) ) + npy# loglikelyhood 
 lvwte<-ptelogs>ntelogs
 gotright<-lvwte==nteby
 tescore[wi]<-sum(gotright)/(sum(gotright)+sum(!gotright))
}

cat ("Accuracy for Part A : ", mean(tescore))



#Question 1 , Part B  #############################################################
#Source Reference : http://luthuli.cs.uiuc.edu/~daf/courses/AML-18/aml-home.html

for (wi in 1:10)
{wtd<-createDataPartition(y=bigy, p=.8, list=FALSE) #create list of indexes for 80% of data
 nbx<-bigx # copy all x's to new list
 
 #Change the 4 columns mentioned in the assignment to NA values
 nbx$BP[nbx$BP == 0] <- NA
 nbx$TSFT[nbx$TSFT == 0] <- NA
 nbx$BMI[nbx$BMI == 0] <- NA
 nbx$AGE[nbx$AGE == 0] <- NA
 
 ntrbx<-nbx[wtd, ] #apply index to x and create new list to filter x to 80% - train
 ntrby<-bigy[wtd] #apply index to y and create new list to filter y to 80% - train
 trposflag<-ntrby>0 # list of y's with TRUE/FLASE for people with diabeties - train
 ptregs<-ntrbx[trposflag, ] # filter x for people with diabeties - train
 ntregs<-ntrbx[!trposflag,] # filter x for people without diabetis- train
 
 #Calculate the P(y) and insert into ppy for positive diabetis and npy for negative diabetes
 #then add ppy and npy to the loglikelyhood based on norm distribution lower down in the code
 ppy <- log(nrow(ptregs)/nrow(ntrbx))
 npy <- log(nrow(ntregs)/nrow(ntrbx))
 
 ntebx<-nbx[-wtd, ] #create test for x
 nteby<-bigy[-wtd] #create test for y
 ptrmean<-sapply(ptregs, mean, na.rm=TRUE) #mean for all features for people with diabeties - trian
 ntrmean<-sapply(ntregs, mean, na.rm=TRUE) #mean for all features for people without diabeties - trian
 ptrsd<-sapply(ptregs, sd, na.rm=TRUE)  #sd for all features for people with diabeties - trian
 ntrsd<-sapply(ntregs, sd, na.rm=TRUE)#sd for all features for people without diabeties - trian
 ptroffsets<-t(t(ntrbx)-ptrmean) #subtract mean from all people, mean is from people with diabeties? 
 ptrscales<-t(t(ptroffsets)/ptrsd)# takes the features after mean subtracting and divides by sd only for diabetis - training
 ptrlogs<--( (1/2)*rowSums(apply(ptrscales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd)) ) + ppy # loglikelyhood 
 ntroffsets<-t(t(ntrbx)-ntrmean) #non dia
 ntrscales<-t(t(ntroffsets)/ntrsd) #non dia sd norm
 ntrlogs<--( (1/2)*rowSums(apply(ntrscales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd)) ) + npy
 lvwtr<-ptrlogs>ntrlogs
 gotrighttr<-lvwtr==ntrby
 trscore[wi]<-sum(gotrighttr)/(sum(gotrighttr)+sum(!gotrighttr))
 pteoffsets<-t(t(ntebx)-ptrmean)
 ptescales<-t(t(pteoffsets)/ptrsd)
 ptelogs<--( (1/2)*rowSums(apply(ptescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd)) ) + ppy
 nteoffsets<-t(t(ntebx)-ntrmean)
 ntescales<-t(t(nteoffsets)/ntrsd)
 ntelogs<--( (1/2)*rowSums(apply(ntescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd)) ) + npy
 lvwte<-ptelogs>ntelogs
 gotright<-lvwte==nteby
 tescore[wi]<-sum(gotright)/(sum(gotright)+sum(!gotright))
}

cat ( "Accuracy for Part B : " , mean(tescore))

#Question 1 , Part C  ############################################################
#Source Reference : http://luthuli.cs.uiuc.edu/~daf/courses/AML-18/aml-home.html

bigx<-wdat[,-c(9)]
bigy<-as.factor(wdat[,9])
wtd<-createDataPartition(y=bigy, p=.8, list=FALSE)
trax<-bigx[wtd,]
tray<-bigy[wtd]
model <- train(trax,as.factor(tray),method = "nb",trControl=trainControl(method='cv',number=10))
#model <- train(ntrbx,as.factor(ntrby),method = "nb",trControl=trainControl(method='cv',number=10))

#Source Reference : https://rpubs.com/maulikpatel/224581
v_predict = predict(model$finalModel,bigx[-wtd,]) # table() gives frequency table, prop.table() gives freq% table.

#Source Reference : #https://stackoverflow.com/questions/40080794/calculating-prediction-accuracy-of-a-tree-using-rparts-predict-method-r-progra

confMat <- table(bigy[-wtd],v_predict$class)
accuracy <- sum(diag(confMat))/sum(confMat)
cat ("Accuracy for Part C: ", accuracy)
cat ("\nConformance Matrix for Part C: ") 
confMat

#Question 1 , Part D  #########################################################
#Source Reference : http://luthuli.cs.uiuc.edu/~daf/courses/AML-18/aml-home.html
cat("Arruracy for Part D: ")
bigx<-wdat[,-c(9)]
bigy<-as.factor(wdat[,9])
wtd<-createDataPartition(y=bigy, p=.8, list=FALSE)
svm<-svmlight(bigx[wtd,], bigy[wtd], pathsvm='C:/Users/Splinta/GoogleDrive/Masters/Courses/CS498/Assignments')
labels<-predict(svm, bigx[-wtd,])
foo<-labels$class
sum(foo==bigy[-wtd])/(sum(foo==bigy[-wtd])+sum(!(foo==bigy[-wtd])))




