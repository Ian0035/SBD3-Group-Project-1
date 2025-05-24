# predict_wages_with_additional_vars.R
# Predicting Wages: A Data-Driven Workflow with Explanations

# -----------------------------
# 1. Load Required Libraries
# -----------------------------
# Load necessary libraries for modeling, explainability, and data handling.
# We use these to ensure we have tools for machine learning, visualization, and manipulation.

if(!require(caret)) install.packages("caret", dependencies = TRUE)
if(!require(randomForest)) install.packages("randomForest", dependencies = TRUE)
if(!require(pdp)) install.packages("pdp", dependencies = TRUE)
if(!require(ggplot2)) install.packages("ggplot2", dependencies = TRUE)
if(!require(dplyr)) install.packages("dplyr", dependencies = TRUE)
if(!require(caretEnsemble)) install.packages("caretEnsemble", dependencies = TRUE)
if(!require(h2o)) install.packages("h2o")
library(h2o)
library(caret)
library(caretEnsemble)
library(randomForest)
library(pdp)
library(ggplot2)
library(dplyr)
library(data.table)

# -----------------------------
# 2. Load the Data
# -----------------------------
# We use the provided data_wage.RData file containing survey results.
load("data_wage.RData")
print(ls()) # Check available objects
df <- data

# -----------------------------
# 3. Data Exploration & Cleaning
# -----------------------------
# Goal: Understand the data before modeling.
# Why? Data-driven decisions start with exploration.

str(df)
summary(df)

# Check for missing values.
missing_summary <- sapply(df, function(x) sum(is.na(x)))
print(missing_summary)

# Decision: Drop rows with missing data if the proportion is low.
cat("Rows before NA removal:", nrow(df), "\n")
df <- na.omit(df)
cat("Rows after NA removal:", nrow(df), "\n")

# Outlier analysis: Spot unusual combinations of high wage and little experience.
df$outlier_flag <- with(df, 
                        (wage > 100000 & (years_experience %in% c("0-1", "1-2") | age %in% c("18-21", "22-24")))
)
cat("High wage/low exp outliers:", sum(df$outlier_flag), "\n")

# Double-check: Are there cases that contradict the outlier logic?
df$disporve_outlier_flag <- with(df, 
                                 (wage < 50000 & (years_experience %in% c("0-1", "1-2") & age %in% c("18-21")) &
                                    ML_atwork %in% c("We have well established ML methods (i.e., models in production for more than 2 years)","We recently started using ML methods (i.e., models in production for less than 2 years)"))
)
cat("Low wage/low exp/ML:", sum(df$disporve_outlier_flag), "\n")

# -----------------------------
# 4. Exploratory Data Analysis (EDA)
# -----------------------------
# Why? To justify feature selection based on data relationships.

# Numeric correlation with wage
numeric_vars <- sapply(df, is.numeric)
if (sum(numeric_vars) > 1) {
  cor_wage <- cor(df[, numeric_vars], use = "complete.obs")
  print(cor_wage["wage", ])
}

# Group means for categorical features
cat_vars <- c("education", "years_experience", "gender", "country", "job_role", "industry", "age")
for (v in cat_vars) {
  print(ggplot(df, aes_string(x = v, y = "wage")) +
          geom_boxplot() +
          ggtitle(paste("Wage by", v)))
}

# Statistical test: Does education level affect wage?
anova_education <- aov(wage ~ education, data = df)
cat("ANOVA for wage by education:\n")
print(summary(anova_education))

# Use these EDA results to select variables showing clear wage differences.

# -----------------------------
# 5. Data Preparation & Encoding
# -----------------------------
# Convert selected variables to factors (needed for modeling and dummy encoding).
df$gender <- as.factor(df$gender)
df$education <- as.factor(df$education)
df$country <- as.factor(df$country)
df$age <- as.factor(df$age)
df$years_experience <- as.factor(df$years_experience)
df$job_role <- as.factor(df$job_role)
df$industry <- as.factor(df$industry)

# Data-driven feature selection:
# We include variables that EDA showed to have strong associations with wage.
model_data <- df %>% 
  select(wage, age, years_experience, education, gender, country, job_role, industry)

# Dummy encoding for categorical predictors (excluding wage).
# Why? Many ML models require numeric input; dummyVars handles this.
dummy_model <- dummyVars(~ ., data = model_data[,-1])
dummy_data <- predict(dummy_model, newdata = model_data[,-1])
model_matrix <- data.frame(wage = model_data$wage, dummy_data)

# -----------------------------
# 6. Feature Importance Pre-check
# -----------------------------
# Run a quick Random Forest to check initial feature importances.
# Why? Data-driven justification for which features to keep.
set.seed(42)
rf_temp <- randomForest(wage ~ ., data = model_matrix, importance = TRUE, ntree = 100)
print(importance(rf_temp))
varImpPlot(rf_temp, main="Initial Feature Importance")

# Optionally: Only keep top features, but here we keep all for interpretability.

# -----------------------------
# 7. Train-Test Split
# -----------------------------
# Why? To estimate how well our model generalizes to new data.
set.seed(123)
train_index <- createDataPartition(model_matrix$wage, p = 0.7, list = FALSE)
train_data <- model_matrix[train_index, ]
test_data  <- model_matrix[-train_index, ]

# -----------------------------
# 8. AutoML Modeling (H2O)
# -----------------------------
# Why? To efficiently compare many models and avoid bias in model selection.

h2o.init(max_mem_size = "2G") # Set max memory as needed

# Convert to H2O frame for AutoML.
df_h2o <- as.h2o(model_matrix)
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

# Show leaderboard and best model
lb <- automl@leaderboard
print(lb)
best_model <- automl@leader

# -----------------------------
# 9. Model Performance Comparison
# -----------------------------
# Why? To select the objectively best-performing model based on test data.

# Performance of AutoML model
perf_best_rmse <- h2o.performance(best_model, valid)
aml_rmse <- h2o.rmse(perf_best_rmse)
aml_mae  <- h2o.mae(perf_best_rmse)
aml_r2   <- h2o.r2(perf_best_rmse)

# Performance of Random Forest (train/test split)
rf_model <- randomForest(wage ~ ., data = train_data, importance = TRUE)
rf_pred <- predict(rf_model, test_data)
rf_rmse <- sqrt(mean((test_data$wage - rf_pred)^2))
rf_mae  <- mean(abs(test_data$wage - rf_pred))
rf_r2   <- cor(test_data$wage, rf_pred)^2

# Results summary table
results_table <- data.frame(
  Model = c("Random Forest", "H2O AutoML"),
  RMSE = c(rf_rmse, aml_rmse),
  MAE  = c(rf_mae, aml_mae),
  R2   = c(rf_r2, aml_r2)
)
print(results_table)

# Data-driven decision: Select the model with lowest RMSE
best_model_name <- results_table$Model[which.min(results_table$RMSE)]
cat("Best performing model:", best_model_name, "\n")

# -----------------------------
# 10. Explainability: Feature Importance & Partial Dependence
# -----------------------------
# Why? To understand *why* the model predicts high/low wages.

# For Random Forest
importance_values <- importance(rf_model)
print(importance_values)
varImpPlot(rf_model, main = "Variable Importance for Wage Prediction (RF)")
cat("Top 5 RF features:\n")
print(head(sort(importance(rf_model)[,1], decreasing = TRUE), 5))

# Partial dependence plot for years_experience
pdp_experience <- partial(rf_model, pred.var = "years_experience", train = train_data)
plotPartial(pdp_experience, main = "Partial Dependence: Wage ~ Years of Experience",
            xlab = "Years of Experience", ylab = "Predicted Wage")

# For H2O AutoML: Explain GBM (often one of the top models)
GBM_rsme <- h2o.get_best_model(automl, algorithm = "GBM", criterion = "rmse")
h2o.varimp_plot(GBM_rsme)

# -----------------------------
# 11. Visualizing Model Predictions
# -----------------------------
# Why? To diagnose errors and check realism.

# AutoML predictions
pred_best_rmse <- h2o.predict(best_model, valid)
pred_df <- as.data.frame(h2o.cbind(pred_best_rmse, valid$wage))
colnames(pred_df) <- c("predicted", "actual")

# Residuals plot
ggplot(pred_df, aes(x = actual, y = predicted - actual)) +
  geom_point(alpha = 0.4) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residuals vs. Actual Wage", x = "Actual Wage", y = "Residuals")

# Predicted vs actual
ggplot(pred_df, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "blue", linetype = "dashed") +
  labs(title = "Predicted vs. Actual Wage", x = "Actual Wage", y = "Predicted Wage")

# -----------------------------
# 12. Real-world Application: Team Member Wage Prediction
# -----------------------------
# Why? To make practical use of our model.

team_raw <- data.frame(
  age = factor(c("30-34", "35-39", "22-24"), levels = levels(df$age)),
  years_experience = factor(c("5-11", "5-11", "0-1"), levels = levels(df$years_experience)),
  education = factor(c("Master’s degree", "Doctoral degree", "Bachelor’s degree"), levels = levels(df$education)),
  gender = factor(c("Female", "Male", "Male"), levels = levels(df$gender)),
  country = factor(c("United States of America", "United States of America", "Switzerland"), levels = levels(df$country)),
  job_role = factor(c("Data Scientist", "Software Engineer", "Student"), levels = levels(df$job_role)),
  industry = factor(c("Computers/Technology", "Computers/Technology", "I am a student"), levels = levels(df$industry))
)

# Dummy encode and predict (using RF for simplicity, or switch to AutoML as needed)
team_dummy <- predict(dummy_model, newdata = team_raw)
team_matrix <- data.frame(team_raw, team_dummy)
team_matrix$predicted_wage <- predict(rf_model, newdata = team_matrix)

team_predictions <- team_matrix[, c("age", "years_experience", "education", "gender", "country", "job_role", "industry", "predicted_wage")]
team_predictions$predicted_wage <- round(team_predictions$predicted_wage, 0)
print(team_predictions)
write.csv(team_predictions, "team_predictions_clean.csv", row.names = FALSE)

# -----------------------------
# 13. Save the Model and Results
# -----------------------------
# Why? For reproducibility and future use.
save(rf_model, file = "rf_wage_model.RData")
write.csv(team_predictions, file = "team_wage_predictions.csv", row.names = FALSE)

# -----------------------------
# END OF SCRIPT
# -----------------------------


