library("dplyr")
library("fastDummies")
library("openxlsx")
library("glmnet")
library('caret')
library("tibble")
library("xgboost")
options(max.print=1000000)

#Reading Data
train_data <- read.csv('train.csv')
test_data <- read.csv('test.csv')

#Data combination and subsequent wrangling
train_test <- rbind(train_data,test_data)

train_test <- dummy_cols(train_test, select_columns = c('PAYMENT_METHOD','CARD_TYPE','USER_STATUS1','USER_STATUS2',
                                                        'PRODUCT_TYPE','SIGNUP_MONTH','USER_HEARD_ABOUT_US',
                                                        'MARKETING_CONTEXT1','MARKETING_CONTEXT2','MARKETING_CONTEXT3',
                                                        'USER_OS','USER_DEVICE'), remove_first_dummy = TRUE)
train_test <- mutate(train_test, Addon_large=case_when(grepl("large",PRODUCT_ADD_ONS,fixed = TRUE)~1,
                                                       TRUE ~ 0),
                     Addon_sb=case_when(grepl("sb",PRODUCT_ADD_ONS,fixed = TRUE)~1,
                                        TRUE ~ 0),
                     Addon_cm=case_when(grepl("cm",PRODUCT_ADD_ONS,fixed = TRUE)~1,
                                        TRUE ~ 0),
                     Addon_abw=case_when(grepl("abw",PRODUCT_ADD_ONS,fixed = TRUE)~1,
                                         TRUE ~ 0),
                     Addon_hc=case_when(grepl("hc",PRODUCT_ADD_ONS,fixed = TRUE)~1,
                                        TRUE ~ 0),
                     Addon_lb=case_when(grepl("lb",PRODUCT_ADD_ONS,fixed = TRUE)~1,
                                        TRUE ~ 0),
                     Addon_crm=case_when(grepl("crm",PRODUCT_ADD_ONS,fixed = TRUE)~1,
                                         TRUE ~ 0),
                     Addon_small=case_when(grepl("small",PRODUCT_ADD_ONS,fixed = TRUE)~1,
                                           TRUE ~ 0),
                     Addon_mmr=case_when(grepl("mmr",PRODUCT_ADD_ONS,fixed = TRUE)~1,
                                         TRUE ~ 0))
train_test$AGE[is.na(train_test$AGE)] <- median(train_test$AGE, na.rm = TRUE) 
train_test$Median_Income[is.na(train_test$Median_Income)] <- median(train_test$Median_Income, na.rm = TRUE) 

train_test <- mutate(train_test, UserAtt1=case_when(grepl("1",USER_ATTRIBUTE1,fixed = TRUE)~1,
                                                    TRUE ~ 0),
                     UserAtt2=case_when(grepl("2",USER_ATTRIBUTE1,fixed = TRUE)~1,
                                        TRUE ~ 0),
                     UserAtt3=case_when(grepl("3",USER_ATTRIBUTE1,fixed = TRUE)~1,
                                        TRUE ~ 0),
                     UserAtt4=case_when(grepl("4",USER_ATTRIBUTE1,fixed = TRUE)~1,
                                        TRUE ~ 0),
                     UserAtt5=case_when(grepl("5",USER_ATTRIBUTE1,fixed = TRUE)~1,
                                        TRUE ~ 0),
                     UserAtt6=case_when(grepl("6",USER_ATTRIBUTE1,fixed = TRUE)~1,
                                        TRUE ~ 0))


train_test <- mutate(train_test, UserAtt7=case_when(grepl("7",USER_ATTRIBUTE2,fixed = TRUE)~1,
                                                    TRUE ~ 0),
                     UserAtt8=case_when(grepl("8",USER_ATTRIBUTE2,fixed = TRUE)~1,
                                        TRUE ~ 0),
                     UserAtt9=case_when(grepl("9",USER_ATTRIBUTE2,fixed = TRUE)~1,
                                        TRUE ~ 0),
                     UserAtt10=case_when(grepl("10",USER_ATTRIBUTE2,fixed = TRUE)~1,
                                         TRUE ~ 0),
                     UserAtt11=case_when(grepl("11",USER_ATTRIBUTE2,fixed = TRUE)~1,
                                         TRUE ~ 0))


train_test <- mutate(train_test, UserAtt12=case_when(grepl("12",USER_ATTRIBUTE3,fixed = TRUE)~1,
                                                     TRUE ~ 0),
                     UserAtt13=case_when(grepl("13",USER_ATTRIBUTE3,fixed = TRUE)~1,
                                         TRUE ~ 0),
                     UserAtt14=case_when(grepl("14",USER_ATTRIBUTE3,fixed = TRUE)~1,
                                         TRUE ~ 0))


train_test <- mutate(train_test, UserAtt16=case_when(grepl("16",USER_ATTRIBUTE4,fixed = TRUE)~1,
                                                     TRUE ~ 0),
                     UserAtt17=case_when(grepl("17",USER_ATTRIBUTE4,fixed = TRUE)~1,
                                         TRUE ~ 0),
                     UserAtt18=case_when(grepl("18",USER_ATTRIBUTE4,fixed = TRUE)~1,
                                         TRUE ~ 0),
                     UserAtt19=case_when(grepl("19",USER_ATTRIBUTE4,fixed = TRUE)~1,
                                         TRUE ~ 0),
                     UserAtt20=case_when(grepl("20",USER_ATTRIBUTE4,fixed = TRUE)~1,
                                         TRUE ~ 0)) 


train_test <- mutate(train_test, UserAtt21=case_when(grepl("21",USER_ATTRIBUTE5,fixed = TRUE)~1,
                                                     TRUE ~ 0),
                     UserAtt22=case_when(grepl("22",USER_ATTRIBUTE5,fixed = TRUE)~1,
                                         TRUE ~ 0),
                     UserAtt23=case_when(grepl("23",USER_ATTRIBUTE5,fixed = TRUE)~1,
                                         TRUE ~ 0),
                     UserAtt24=case_when(grepl("24",USER_ATTRIBUTE5,fixed = TRUE)~1,
                                         TRUE ~ 0))

train_test = subset(train_test, select = -c(3:8,10:20,22:25))

test <- train_test[is.na(train_test$LTV),]
train <- train_test[!is.na(train_test$LTV),]

test %>% add_column(LTV = NA)

#Dividing train set into "train" and "validation" sets
dt = sort(sample(nrow(train), nrow(train)*.7))
train_1<-train[dt,]
test_1<-train[-dt,]

#define predictor and response variables in training set
train_1_x = data.matrix(train_1[, -c(1,5)])
train_1_y = train_1[,5]

#define predictor and response variables in validation set
test_1_x = data.matrix(test_1[, -c(1,5)])
test_1_y = test_1[, 5]
test_1_y[is.na(test_1_y)] <- 0 

#fit XGBoost model to training (& validation) set
xgb_train = xgb.DMatrix(data = train_1_x, label = train_1_y)
xgb_test = xgb.DMatrix(data = test_1_x, label = test_1_y)

#define watchlist
watchlist = list(train_1=xgb_train, test_1=xgb_test)

#Intermittent testing codes from line 126 to 131
#params <- list(booster = "gbtree", objective = "reg:squarederror", 
#               eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)
#xgbcv <- xgb.cv( params = params, data = xgb_train, nrounds = 100, nfold = 5, showsd = T,
#                stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)
#create hyperparameter grid
#hyper_grid <- expand.grid(max_depth = seq(2, 3, 1), eta = seq(.1, .3, .05)) 

#fit XGBoost model and display training and testing data at each round
model = xgb.train(data = xgb_train, max.depth = 5, watchlist=watchlist, nrounds = 1000,
                  early_stopping_rounds=50 ,objective = "reg:squarederror",
                  eta=0.1, subsample=1,colsample_bytree=0.5, alpha=5, lambda=25,gamma=50,
                  min_child_weight=1)

#define final model
final = xgboost(data = xgb_train, max.depth = 5, nrounds = 1000, verbose = 0, early_stopping_rounds=50 ,
                objective = "reg:squarederror", eta=0.1, subsample=1,colsample_bytree=0.5,
                alpha=5, lambda=25,gamma=50,min_child_weight=1)

#use model to make predictions on test data
pred_y = predict(final, xgb_test)
pred_y[pred_y<0]<-0
pred <- cbind(test,as.data.frame(pred_y))
write.xlsx(pred[,c(1,1471)], 'predict_xgboost.xlsx')

#measure prediction accuracy
mean((test_1_y - pred_y)^2) #mse
caret::MAE(test_1_y, pred_y) #mae
caret::RMSE(test_1_y, pred_y) #rmse

#Finally, this model was used to make predictions on the actual provided test data and the rmse was measured by Kaggle
