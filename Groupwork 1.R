# predict_wages_with_additional_vars.R

# -------------------------
# 1. Load required libraries
# -------------------------
if(!require(caret)) install.packages("caret", dependencies = TRUE)
if(!require(randomForest)) install.packages("randomForest", dependencies = TRUE)
if(!require(pdp)) install.packages("pdp", dependencies = TRUE)
if(!require(ggplot2)) install.packages("ggplot2", dependencies = TRUE)
if(!require(dplyr)) install.packages("dplyr", dependencies = TRUE)
if(!require(caretEnsemble)) install.packages("caretEnsemble", dependencies = TRUE)
if(!require(h2o)) install.packages("h2o")
# Remove old h2o first if installed
if ("package:h2o" %in% search()) detach("package:h2o", unload=TRUE)
if ("h2o" %in% rownames(installed.packages())) remove.packages("h2o")

# Install dependencies
install.packages(c("RCurl","jsonlite"))

# Install latest stable H2O package (replace the URL below if needed)
install.packages("h2o", 
                 repos = "https://h2o-release.s3.amazonaws.com/h2o/latest_stable_R")



library(h2o)
library(caret)
library(caretEnsemble)
library(randomForest)
library(pdp)
library(ggplot2)
library(dplyr)
library(data.table)


# -------------------------
# 2. Load the Data
# -------------------------
load("data_wage.RData")
print(ls())
df <- data

# -------------------------
# 3. Data Exploration & Preprocessing
# -------------------------
str(df)
summary(df)
View(df)

missing_summary <- sapply(df, function(x) sum(is.na(x)))
print(missing_summary)

df <- na.omit(df)

#Looking for potential outliers, people who are very young and earn more than 100k: conclusion from this is ML_Atwork might be important.
df$outlier_flag <- with(df, 
                        (wage > 100000 & (years_experience %in% c("0-1", "1-2") | age %in% c("18-21", "22-24")))
)

View(df[df$outlier_flag == TRUE, ])

#We can't make direct conclusions from this as there are plenty of people who earn less than 50k and also use ML at work.
df$disporve_outlier_flag <- with(df, 
                        (wage < 50000 & (years_experience %in% c("0-1", "1-2") & age %in% c("18-21") & ML_atwork %in% c("We have well established ML methods (i.e., models in production for more than 2 years)","We recently started using ML methods (i.e., models in production for less than 2 years)")))
)
View(df[df$disporve_outlier_flag == TRUE, ])


# Convert key variables to factors
df$gender <- as.factor(df$gender)
df$education <- as.factor(df$education)
df$country <- as.factor(df$country)
df$age <- as.factor(df$age)
df$years_experience <- as.factor(df$years_experience)
df$job_role <- as.factor(df$job_role)
df$industry <- as.factor(df$industry)
# -------------------------
# 4. Feature Selection and Dummy Encoding
# -------------------------
# Include wage, age, years_experience, education, gender, country, job_role, and industry.
model_data <- df %>% 
  select(wage, age, years_experience, education, gender, country, job_role, industry)

# Dummy encoding for all predictors (excluding wage)
dummy_model <- dummyVars(~ ., data = model_data[,-1])
dummy_data <- predict(dummy_model, newdata = model_data[,-1])
model_matrix <- data.frame(wage = model_data$wage, dummy_data)

# -------------------------
# 5. Train-Test Split
# -------------------------
set.seed(123)
train_index <- createDataPartition(model_matrix$wage, p = 0.7, list = FALSE)
train_data <- model_matrix[train_index, ]
test_data  <- model_matrix[-train_index, ]
# -------------------------
# 6.0 Trying to implement AutoML
# -------------------------
h2o.init()
h2o.xgboost.available()

# Convert df to H2OFrame
df_h2o <- as.h2o(model_matrix)  # model_matrix includes dummy-encoded predictors
set.seed(12)
splits <- h2o.splitFrame(df_h2o, ratios = 0.8, seed = 1234)
train <- splits[[1]]
valid <- splits[[2]]

dep_var <- "wage"
indep_vars <- setdiff(colnames(df_h2o), dep_var)

automl <- h2o.automl(
  x = indep_vars,
  y = dep_var,
  training_frame = train,
  leaderboard_frame = valid,
  max_models = 10,
  seed = 12,
  sort_metric = "RMSE",
  exclude_algos = c("XGBoost")
)


lb <- automl@leaderboard
print(lb)

best_model <- automl@leader
print(best_model)

# Find the best performing model per a certain criteria and explore it.
best_rmse <- h2o.get_best_model(automl, criterion = "RMSE")   # Best model per the rmse indicator. 
best_rmse                                                   # Let's explore the best performing model 

pred_best_rmse <- h2o.predict(best_rmse, valid)
predictions <- as.data.table(pred_best_rmse)
perf_best_rmse <- h2o.performance(best_rmse, valid)
performance_single <- function(perf_object) {
  rmse <- h2o.rmse(perf_object)
  mse <- h2o.mse(perf_object)
  mae <- h2o.mae(perf_object)
  r2  <- h2o.r2(perf_object)
  
  cat("Performance Metrics:\n")
  cat("---------------------\n")
  cat(sprintf("RMSE: %.2f\n", rmse))
  cat(sprintf("MSE : %.2f\n", mse))
  cat(sprintf("MAE : %.2f\n", mae))
  cat(sprintf("R²  : %.4f\n", r2))
}

# Check the results for the best performing model (by rmse value)
single_best_rmse <- performance_single(perf_best_rmse)

# Extract predictions and actuals
pred_df <- as.data.frame(h2o.cbind(pred_best_rmse, valid$wage))
colnames(pred_df) <- c("predicted", "actual")

# Residual plot
ggplot(pred_df, aes(x = actual, y = predicted - actual)) +
  geom_point(alpha = 0.4) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residuals vs. Actual Wage", x = "Actual Wage", y = "Residuals")

#Predicted vs actual plot
ggplot(pred_df, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "blue", linetype = "dashed") +
  labs(title = "Predicted vs. Actual Wage", x = "Actual Wage", y = "Predicted Wage")


# -------------------------
# 6.2 AutoML explanibility
# -------------------------

# Explain leader model & compare with all AutoML models
exp_automl <- h2o.explain(automl, valid) 
print(exp_automl) 

StackedEnsemble_rsme <- h2o.get_best_model(automl, algorithm = "StackedEnsemble", criterion = "rmse")
GBM_rsme <- h2o.get_best_model(automl, algorithm = "GBM", criterion = "rmse")

exp_GBM <- h2o.explain(GBM_rsme, valid)  # Explain the best performing GBM model 
print(exp_GBM)

h2o.varimp_plot(GBM_rsme)

# -------------------------
# 6. Model Development
# -------------------------
set.seed(123)
rf_model <- randomForest(wage ~ ., data = train_data, importance = TRUE)
print(rf_model)

# -------------------------
# 7. Model Evaluation
# -------------------------
predictions <- predict(rf_model, test_data)
rmse_value <- sqrt(mean((test_data$wage - predictions)^2))
rsq_value <- cor(test_data$wage, predictions)^2

cat("Test RMSE:", rmse_value, "\n")
cat("Test R-squared:", rsq_value, "\n")

# -------------------------
# 8. Explainability: Feature Importance & Partial Dependence
# -------------------------
importance_values <- importance(rf_model)
print(importance_values)
varImpPlot(rf_model, main = "Variable Importance for Wage Prediction")

pdp_experience <- partial(rf_model, pred.var = "years_experience", train = train_data)
plotPartial(pdp_experience, main = "Partial Dependence of Wage on Years of Experience",
            xlab = "Years of Experience", ylab = "Predicted Wage")

# -------------------------
# 9. Real-world Application: Predicting Future Wages for Team Members
# -------------------------
# Define raw inputs for team members.
# Adjust these values to match your dataset's factor levels.
team_raw <- data.frame(
  age = factor(c("30-34", "35-39", "22-24"), levels = levels(df$age)),
  years_experience = factor(c("5-11", "5-11", "0-1"), levels = levels(df$years_experience)),
  education = factor(c("Master’s degree", "Doctoral degree", "Bachelor’s degree"), levels = levels(df$education)),
  gender = factor(c("Female", "Male", "Male"), levels = levels(df$gender)),
  country = factor(c("United States of America", "United States of America", "Switzerland"), 
                   levels = levels(df$country)),
  job_role = factor(c("Data Scientist", "Software Engineer", "Student"), levels = levels(df$job_role)),
  industry = factor(c("Computers/Technology", "Computers/Technology", "I am a student"), levels = levels(df$industry))
)

# Apply the same dummy encoding
team_dummy <- predict(dummy_model, newdata = team_raw)
team_matrix <- data.frame(team_raw, team_dummy)

# Predict wages for team members
team_matrix$predicted_wage <- predict(rf_model, newdata = team_matrix)

print(team_matrix[, c("age", "years_experience", "education", "gender", "country", "job_role", "industry", "predicted_wage")])

# -------------------------
# 10. Save the Model and Results (Optional)
# -------------------------
save(rf_model, file = "rf_wage_model.RData")
write.csv(team_matrix, file = "team_wage_predictions.csv", row.names = FALSE)

# -------------------------
# End of Script
