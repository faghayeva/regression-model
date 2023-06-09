# Linear regression model with mlr3data::kc_housing
# importing kc_housing dataset from ml3data package. ----
library(tidyverse)
library(data.table)
library(inspectdf)
library(dplyr)
library(mice)
library(recipes)
library(graphics)
library(caret)
library(h2o)
library(Metrics)
library(plotly)
library(glue)
library(patchwork)
library(mlr3data)
localH2O = h2o.init()
demo(h2o.kmeans)

data <- mlr3data::kc_housing

# data cleaning - resolving NA values----
data %>% glimpse()
data %>% inspect_na()

data %>% 
  inspect_na() %>% 
  filter(pcnt<40) %>% 
  pull(col_name) -> variables

data <- data %>% select(all_of(variables))

# Resolving outliers----

df.num <- data %>% 
  select_if(is.numeric) %>% 
  select(all_of(target), everything())

target <- "price"

solve_outliers <- function(data, target){
  num_vars <- data %>% 
  select(-all_of(target)) %>% 
  names()
  
  for_vars <- c()
  for (b in 1:length(num_vars)){
    OutVals <- boxplot(data[[num_vars[b]]], plot =F)$out
    if(length(OutVals)>0){
      for_vars[b] <- num_vars[b]
    }
  }
  for_vals <- for_vals %>% as.data.frame() %>% drop_na() %>%pull(.)  
  for_vals %>% length()
  
for (o in for_vars){
  OutVals <- boxplot(data[[o]], plot-F)$out
  mean <- mean(data[[o]], na.rm=T)
  
  o3 <- ifelse(OutVals>mean, OutVals, NA) %>% na.omit() %>% as.matrix() %>% .[,1]
  o1 <- ifelse(OutVals<mean, OutVals, NA) %>% na.omit() %>% as.matrix() %>% .[,1]
  
  val3 <- quantile(data[[o]], 0.75, na.rm=T)+1.5*IQR(data[[o]], na.rm=T)
  data[which(data[[o]]%in%o3),o]<-val3
  
  val1 <- quantile(data[[o]], 0.75, na.rm=T)-1.5*IQR(data[[o]], na.rm=T)
  data[which(data[[o]]%in%o3),o]<-val1
  
 }
  return(data)
}  

df.num <- df.num %>% solve_outliers(target = target)

#  Applying "One Hote Encoding" ----

df.chr <- data %>% 
  mutate_if(is.factor,as.character) %>% 
  select_if(is.character)

df.chr <- dummyVars(" ~ .", df.chr ) %>% 
  predict(df.chr) %>% 
  as.data.frame()

df <- cbind(df.chr,df.num) %>% 
  select(all_of(target), everything())
# data doesn't have character variables, that's why there's no need one hote encoding

# Resolving "Multicollinearity" problem----

df <- data %>% as.data.frame()

features <- df %>% select(-all_of(target)) %>% names()

f <- as.formula(paste(target, paste(features, collapse = "+"), sep = "~"))
glm <- glm(f, data = df)

glm %>% summary()

coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]
features <- features[!features%in% coef_na]

f <- as.formula(paste(target, paste(features, collapse = "+"), sep = "~"))
glm <- glm(f, data = df)
                
glm %>% summary()
# vif
install.packages("faraway")
library(faraway)
while(glm %>% faraway::vif() %>% sort(decreasing = T) %>% .[1]>=2){
  afterVIF <- glm %>% faraway::vif() %>% sort(decreasing = T) %>% .[1] %>% names()
  f <- as.formula(paste(target, paste(afterVIF, collapse = "+"), sep = "~"))
  glm <- glm(f, data =df)
}

glm %>% faraway::vif() %>% sort(decreasing = T) %>% names()->features

# Standardize variables----

df %>% glimpse()

df[,-1] <- df[,-1] %>% scale() %>% as.data.frame()

# Building GLM (Generalized Linear Model)  with Cross Validation ----

model <- h2o.glm(
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  nfolds = 10, seed = 13,
  lambda = 0, compute_p_values = TRUE
)

# Showing performance of model ----
y_pred <- model %>% h2o.predict(test) %>% as.data.frame()
pred <- y_pred$predict
actual <- test %>% as.data.frame() %>% pull(all_of(target))

eval_func <- function(x, y) summary(lm(y-x))
eval_sum <- eval_func(actual, pred)

eval_sum$adj.r.squared %>% round(2)
mae(actual, pred) %>% round(1)
rmse(actual, pred) %>% round(1)

#plot 
results <- cbind(pred, actual) %>% 
  as.data.frame()

adjusted_r2 <- eval_sum$adj.r.squared

g < results %>%
  ggplot(aes(pred, actual)) +
  geom_point(color = "darkred") +
  geom_smooth(method = lm) +
  labs(x = "Proqnoz dəyərlər",
       y = "Əsl dəyərlər",
       title = glue "Test: Adjusted R2 = {round(enexpr(Adjusted_R2), 2)}") +
  theme(plot.title= element_text(color = "darkgreen", size=16, hjust=0.5),
        axis.text.y = element_text(size=12),
        axis.text.x = element_text(size=12),
        axis.title.x = element_text(size=14),
        axis.title.y = element_text(size=14))

g %>% ggplotly()

# Checking "Overfitting" and "Underfitting" ----

y_pred_train <- model %>% h2o.predict(train) %>% as.data.frame()
pred_train <- y_pred_train$predict
actual_train <- train %>% as.data.frame() %>% pull(all_of(target)
                                                   
eval_sum <- eval_func(actual_train, pred_train)
eval_sum$adj.r.squared %>% round(2)
mae(actual_train, pred_train) %>% round(1)
rmse(actual_train, pred_train) %>% round(1)

# Plot ----
results_train <- cbind(pred_train, actual_train) %>%
  as.data.frame()

Adjusted_R2_train <- eval_sum$adj.r.squared

g_train <- results_train %>%
  ggplot(aes(pred_train, actual_train)) +
  geom_point (color = "darkred") +
  geom_smooth(method = 1m) +
  labs(x = "Proqnoz dəyərlər",
       y = "Əsl dəyərlər",
       title = glue("Train: Adjusted R2 = {round(enexpr(Adjusted_R2_train), 2)}")) +
  theme(plot.title = element_text(color="darkgreen", size=16, hjust=0.5),
        axis.text.y = element_text(size=12),
        axis.text.x = element_text(size=12),
        axis.title.x = element_text(size=14),
        axis.title.y = element_text(size=14))

g_train %>% ggplotly()

# Comparison ----

g_train + g

tibble(Adjusted_R2_train,
       Adjusted_R2_test = adjusted_R2)

