data<-read.csv(file.choose())
View(data)
library(caret)
library(e1071)
library(class)  
library(rpart)  
library(rpart.plot) 
library(mlbench)  
library(ggplot2)

data <- data[,-1]  
data$diagnosis <- ifelse(data$diagnosis == "M", 1, 0)  


set.seed(42)
index <- createDataPartition(data$diagnosis, p = 0.7, list = FALSE)
train_data <- data[index, ]
test_data <- data[-index, ]


X_train <- train_data[,-1]
y_train <- train_data$diagnosis
X_test <- test_data[,-1]
y_test <- test_data$diagnosis


model_dt <- train(factor(diagnosis) ~ ., data = train_data, method = "rpart")
predictions_dt <- predict(model_dt, newdata = test_data)
metrics_dt <- confusionMatrix(predictions_dt, factor(y_test))


model_knn <- train(factor(diagnosis) ~ ., data = train_data, method = "knn", tuneGrid = data.frame(k = 5))
predictions_knn <- predict(model_knn, newdata = test_data)
metrics_knn <- confusionMatrix(predictions_knn, factor(y_test))


model_svm <- train(factor(diagnosis) ~ ., data = train_data, method = "svmLinear")
predictions_svm <- predict(model_svm, newdata = test_data)
metrics_svm <- confusionMatrix(predictions_svm, factor(y_test))


results <- data.frame(
  Model = c("Decision Tree", "KNN", "SVM"),
  Accuracy = c(metrics_dt$overall["Accuracy"], metrics_knn$overall["Accuracy"], metrics_svm$overall["Accuracy"]),
  Precision = c(metrics_dt$byClass["Pos Pred Value"], metrics_knn$byClass["Pos Pred Value"], metrics_svm$byClass["Pos Pred Value"]),
  Recall = c(metrics_dt$byClass["Sensitivity"], metrics_knn$byClass["Sensitivity"], metrics_svm$byClass["Sensitivity"]),
  F1_Score = c(2 * (metrics_dt$byClass["Pos Pred Value"] * metrics_dt$byClass["Sensitivity"]) / (metrics_dt$byClass["Pos Pred Value"] + metrics_dt$byClass["Sensitivity"]),
               2 * (metrics_knn$byClass["Pos Pred Value"] * metrics_knn$byClass["Sensitivity"]) / (metrics_knn$byClass["Pos Pred Value"] + metrics_knn$byClass["Sensitivity"]),
               2 * (metrics_svm$byClass["Pos Pred Value"] * metrics_svm$byClass["Sensitivity"]) / (metrics_svm$byClass["Pos Pred Value"] + metrics_svm$byClass["Sensitivity"]))
)


results_long <- reshape2::melt(results, id.vars = "Model")

ggplot(results_long, aes(x = Model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Performance Comparison", x = "Model", y = "Score") +
  scale_fill_manual(values = c("Accuracy" = "skyblue", "Precision" = "lightgreen", "Recall" = "orange", "F1_Score" = "salmon")) +
  theme_minimal()


ggplot(results_long, aes(x = value, fill = Model)) +
  geom_histogram(bins = 10, position = "dodge", alpha = 0.7) +
  facet_wrap(~ variable, scales = "free") +
  labs(title = "Histogram of Model Performance Metrics", x = "Metric Score", y = "Frequency") +
  scale_fill_manual(values = c("skyblue", "lightgreen", "salmon")) +
  theme_minimal()




