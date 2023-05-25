# Packages Needed
library(caret)
library(xgboost)
library(ggplot2)

# Load Dataset
data <- read.csv('dataset.csv')
# Removes Nulls
data <- na.omit(data)
# View
head(data)

# Split the data into X and Y
set.seed(40) # Random seed for reproducibility
index <- createDataPartition(data$call_for_interview, p = 0.8, list = FALSE)
x_train <- data[index, -ncol(data)]
x_test <- data[-index, -ncol(data)]
y_train <- data[index, ncol(data)]
y_test <- data[-index, ncol(data)]

# Convert Data to Matrix
x_train_matrix <- as.matrix(x_train)
x_test_matrix <- as.matrix(x_test)

# Define the hyperparameters
learning_rate <- 0.15
n_estimators <- 200
max_depth <- 4

# Initiate the XGBoost classifier model
model <- xgboost(data = x_train_matrix, label = y_train,
                 nrounds = n_estimators, 
                 eta = learning_rate, 
                 max_depth = max_depth, 
                 objective = "binary:logistic")

# Create the DMatrix for training and testing
dtrain <- xgb.DMatrix(data = x_train_matrix, label = y_train)
dtest <- xgb.DMatrix(data = x_test_matrix, label = y_test)

# Define the evaluation set
eval_set <- list(test = dtest)
# Define the evaluation metric
eval_metric <- 'logloss'

# Fit the XGBoost classifier model
model <-xgb.train(params = list(eta = learning_rate, max_depth = max_depth),
                  data = dtrain, nrounds = n_estimators,
                  watchlist = eval_set, early_stopping_rounds = 50,
                  eval_metric = eval_metric)

# Create the DMatrix for testing
dtest <- xgb.DMatrix(data = x_test_matrix) 

# Obtain predictions from the model
y_pred <- predict(model, dtest)
# Round the predicted values to integers
predictions <- round(y_pred)

# Create confusion matrix
cm <- table(predictions, y_test)

# Calculate accuracy
accuracy <- sum(diag(cm)) / sum(cm)
# Print accuracy
cat("Accuracy:", sprintf("%.2f%%", accuracy * 100))

# Extract feature importance from the model
importance_df <- xgb.importance(model = model)
# Descending Order
importance_df <- importance_df[order(importance_df$Gain, decreasing = TRUE), ]
# Reorder of Levels
importance_df$Feature <- factor(importance_df$Feature, levels = importance_df$Feature)

# Plot
ggplot(data = importance_df, aes(x = Feature, y = Gain)) +
  geom_bar(stat = 'identity', fill = 'steelblue') +
  geom_text(aes(label = round(Gain, 2)), vjust = -0.5) +
  labs(x = 'Feature', y = 'Importance') +
  ggtitle('Feature Importance') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))