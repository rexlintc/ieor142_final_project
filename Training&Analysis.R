library(dlookr)
library(dplyr)
library(magrittr)
library(ggplot2)
library(ggmosaic)
library(caret)
library(e1071)
library(xgboost)
library(randomForest)
library(keras)
library(doParallel)
library(data.table)
source("C:/methods/R-source/cairo.R")
options(stringsAsFactors = F)


# ===============================================================================
# Data Cleaning and Exploratory Data Analysis
# ===============================================================================


data <- read.csv("_data/train.csv") 
data %<>% 
  mutate(inning = as.numeric(inning), 
         pfx_x = as.numeric(pfx_x) %>% round(3),
         pfx_z = as.numeric(pfx_z) %>% round(3))

data %<>%  mutate( count_na = rowSums(is.na(.)))
table(data$count_na)

data %<>% filter(count_na < 7)

summary(data$pfx_x)
data[is.na(data$pfx_x), "pfx_x"] <- mean(data$pfx_x, na.rm = T)
summary(data$pfx_x)
 
find_na(data, index = F)
data %<>% select(-count_na)

data %<>% filter(type %in% c("CH", "CU", "FA", "FC", "FS", "SI", "SL"))

data$type_literal <- recode(data$type, FA = "Fastball", CU = "Curveball", CH = "Changeup",
                            SI = "Sinker", SL = "Slider", FC = "Fastball Cutter", FS = "Fastball Splitter")

data %<>% select(-y55)

plot_outlier(data)
skewness(data$plate_z)
kurtosis(data$plate_z)

data %>% filter(plate_z < 0) %>% nrow()

fig <- ggplot(data=data, aes(x = type_literal, fill=type_literal)) + geom_bar() + theme_bw() +
  labs(x = "Pitch Type", y = "Count", fill = "Pitch Type")
cairo_plot(fig, "_figs/class freqs.png")

data$type.f <- as.factor(data$type)
data$pitcher_side <- recode(data$pitcher_side, Right = 1, Left = 0)
data$batter_side <- recode(data$batter_side, Right = 1, Left = 0)

saveRDS(data, "_data/data.RDS") 

test_data <- read.csv("_data/test.csv") 
test_data %<>%  mutate( count_na = rowSums(is.na(.)))
table(test_data$count_na)
test_data$count_na <- NULL

plot_outlier(test_data)

test_data$pitcher_side <- recode(test_data$pitcher_side, Right = 1, Left = 0)
test_data$batter_side <- recode(test_data$batter_side, Right = 1, Left = 0)

saveRDS(test_data, "_data/test_data.RDS") 

data <- readRDS("_data/data.RDS") 

png("_figs/correlation_plot.png", width = 960, height = 960, res = 144, pointsize = 8)
plot_correlate(data)
dev.off()

featurePlot(x = select(data, x55:plate_z), 
            y = data$type.f, plot = "box",
            scales = list(x = list(relation = "free"), 
                        y = list(relation = "free")),
            auto.key = list(columns = 6),
            layout = c(5, 6))

data %>% mutate(pitcher_side = ifelse(pitcher_side == 1, "Right", "Left"), 
                batter_side = ifelse(batter_side == 1, "Right", "Left")) %>%
  				ggplot() + geom_mosaic(aes(x = product(pitcher_side, type_literal ), 
                fill = type_literal), na.rm = TRUE, divider = mosaic("v")) +
    			labs(x = "Pitcher Side", title = "Pitch Type by Pitcher Side", y = "Pitch Type")

data %>% mutate(pitcher_side = ifelse(pitcher_side == 1, "Right", "Left"), 
                batter_side = ifelse(batter_side == 1, "Right", "Left")) %>%
  				ggplot() + geom_mosaic(aes(x = product(batter_side, type_literal ), 
                fill=type_literal), na.rm = TRUE, divider = mosaic("v")) +
  				labs(x = "Batter Side", title = "Pitch Type by Batter Side", y = "Pitch Type")

  
# ===============================================================================
# Feature Selection and Modeling
# =============================================================================


clust <- makePSOCKcluster(4)
registerDoParallel(clust)
trellis.par.set(caretTheme())

data <- readRDS("_data/data.RDS")
data <- data[complete.cases(data),]
train_x <- data %>% select(-type, -type.f, -c(pitcher, pitch_id, type_literal))
train_y <- data[, "type.f"]

ctrl <- sbfControl(functions = rfSBF, method = "repeatedcv", repeats = 5)
set.seed(1234)
SBF_rf <- sbf(train_x, train_y, sbfControl = ctrl, multivariate= T )

SBF_rf
SBF_rf %>% predictors()

ctrl <- sbfControl(functions = nbSBF, method = "repeatedcv", repeats = 5)
set.seed(1234)
SBF_nb <- sbf(train_x, train_y, sbfControl = ctrl)
SBF_nb

control <- rfeControl(functions = nbFuncs, method = "cv", number = 5)
RFE_nb <- rfe(train_x, train_y, sizes = c(1:20), rfeControl = control)
saveRDS(RFE_nb, "_models/RFE_nb")

RFE_nb %>% predictors()
png("_figs/Naive Bayes feature selection.png")
plot(RFE_nb, type = c("g", "o"))
dev.off()
RFE_nb$variables  

ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
RFE_rf <- rfe(train_x, train_y, sizes = c(1:20), rfeControl = ctrl)
saveRDS(RFE_rf, "_models/RFE_rf")

png("_figs/Random Forest feature selection.png")
plot(RFE_rf, type = c("g", "o"))
dev.off()

RFE_rf$fit
RFE_rf %>% predictors()

plot(RFE_rf, type = c("g", "o"))
RFE_rf$fit  

formula <- as.formula("train_y ~  pitch_speed + vy55 + x55 + release_x + break_z + extension +       
                       release_z + pfx_zLONG + z55 + ax + pfx_x + pfx_xLONG+       
                       break_x +  approach_angle_x + inning" )

mod_rf <- randomForest(formula, data = cbind(train_x, train_y))

mod_rf

# Plot variable importances
fig <- mod_rf$importance %>% round(1) %>% as.data.frame() %>% mutate(Feature = row.names(.)) %>% 
  mutate(Feature = factor(Feature, levels = Feature[order(MeanDecreaseGini)])) %>%
  ggplot(aes(x = Feature, y = MeanDecreaseGini, fill=Feature)) + geom_bar(stat = "identity"  ) + 
  theme_bw() + theme(legend.position = "none") +
  labs(x = "Variable", y = "Relative Variable Importance", fill = "Feature", title = "Variable Importances") + 
  coord_flip()
  
cairo_plot(fig, "_figs/Variable Importances_rf.png")
saveRDS(mod_rf, "_models/mod_rf.RDS")

data <- readRDS("_data/data.RDS") %>% setDT()
one_hot_encoder <- dummyVars(" ~ type", data = data)  
saveRDS(one_hot_encoder, "_models/one_hot_encoder.RDS")

cols_to_standardize <- names(select(data, inning:plate_z))
means <- data[,lapply(.SD, mean), .SDcols = cols_to_standardize ] %>% 
  as.numeric() %>% setNames(cols_to_standardize)
sds <-   data[,lapply(.SD, sd),   .SDcols = cols_to_standardize]  %>% 
  as.numeric() %>% setNames(cols_to_standardize)
saveRDS(list(means,sds), "_data/std_encoding.RDS")

set.seed(1234)
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))

train_x <- data[ind == 1, ] %>% 
  select(-type, -type.f, -c(pitcher, pitch_id, type_literal))

train_x[, (cols_to_standardize) := sweep(.SD, 2, means, FUN = "-"), .SDcols = cols_to_standardize]
train_x[, (cols_to_standardize) := sweep(.SD, 2, sds, FUN = "/"), .SDcols = cols_to_standardize]
train_y <- data[ind == 1] %>% predict(one_hot_encoder, newdata = .) 

train_x %>% summarize_at(cols_to_standardize, funs(mean,sd)) 
train_x %<>% as.matrix()

val_x <- data[ind == 2, ] %>% 
  select(-type, -type.f, -c(pitcher, pitch_id, type_literal))

val_x[, (cols_to_standardize) := sweep(.SD, 2, means, FUN= "-"), .SDcols = cols_to_standardize]
val_x[, (cols_to_standardize) := sweep(.SD, 2, sds, FUN = "/"), .SDcols = cols_to_standardize]
val_y <- data[ind == 2] %>% predict(one_hot_encoder, newdata = .)

val_x %>% summarize_at(cols_to_standardize, funs(mean,sd)) 
val_x %<>% as.matrix()

model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(34)) %>%
  layer_dropout(.5) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(.5) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 7, activation = "softmax")

model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  train_x,
  train_y,
  epochs = 2000,
  batch_size  = 512,
  validation_data = list(val_x, val_y)
)

save_model_hdf5(model, "_models/nn_01.h5")

model <- load_model_hdf5("_models/nn_01.h5")

model %>% evaluate(train_x, train_y)
model %>% evaluate(val_x, val_y)

pred_train <- model %>% predict(train_x) %>% round(2) %>% as.data.frame() %>% 
  setNames(dimnames(val_y)[[2]] %>% substr(5, 6))
pred_train$type.f <- apply(pred_train, 1, function(x) names(pred_train)[which.max(x)]) %>% as.factor()
confusionMatrix(data[ind == 1, type.f], pred_train$type.f)

pred_val <- model %>% predict(val_x) %>% round(2) %>% as.data.frame() %>% 
  setNames(dimnames(val_y)[[2]] %>% substr(5, 6)) 
pred_val$type.f <- apply(pred_val, 1, function(x) names(pred_val)[which.max(x)]) %>% as.factor()
confusionMatrix(data[ind == 2, type.f], pred_val$type.f)

confusionMatrix(data[ind == 1, "type.f"], pred_train)


# ====================================================================================================
# Predicting Test Set
# ====================================================================================================


library(openxlsx)

# Random Forest predictions
test_x <- read.xlsx("_data/test.xlsx", sheet = "test")
test_x$pitcher_side <- recode(test_x$pitcher_side, Right = 1, Left = 0)
test_x$batter_side <- recode(test_x$batter_side, Right = 1, Left = 0)
test_x$y55 <- NULL

mod_rf <- readRDS("_models/mod_rf.RDS")
test_x$pred_rf <- predict(mod_rf, test_x)
pred_rf <- select(test_x, test_id, pred_rf)

# Neural Net predictions
test_x <- read.xlsx("_data/test.xlsx", sheet = "test") %>% setDT()
test_x$pitcher_side <- recode(test_x$pitcher_side, Right = 1, Left = 0)
test_x$batter_side <- recode(test_x$batter_side, Right = 1, Left = 0)
test_x$y55 <- NULL

cols_to_standardize <- names(select(test_x, inning:plate_z))
stdVals <- readRDS("_data/std_encoding.RDS")
means <- stdVals[[1]]
sds   <- stdVals[[2]]

test_x[, (cols_to_standardize) := sweep(.SD, 2, means, FUN = "-"), .SDcols = cols_to_standardize]
test_x[, (cols_to_standardize) := sweep(.SD, 2, sds, FUN = "/"), .SDcols = cols_to_standardize]

mod_nn <- load_model_hdf5("_models/nn_01.h5")

one_hot_encoder <- readRDS("_models/one_hot_encoder.RDS")
types <- data.frame(type = c("CH", "CU", "FA", "FC", "FS", "SI", "SL")) %>% 
  predict(one_hot_encoder, newdata = .) 

pred_nn <- test_x %>% select(-c(test_id, pitcher, pitch_id)) %>% as.matrix() %>%
  predict(mod_nn, .) %>% round(2) %>% as.data.frame() %>% 
  setNames(dimnames(types)[[2]] %>% substr(5, 6)) 
test_x$pred_nn <- apply(pred_nn, 1, function(x) names(pred_nn)[which.max(x)]) %>% as.factor()
predictions <- cbind(pred_rf, test_x$pred_nn) %>% 
  setNames(c("test_id", "Random Forest Prediction", "Neural Net Prediction"))

wb <- loadWorkbook("_data/test.xlsx")
addWorksheet(wb, "Predictions")
writeData(wb, "Predictions", predictions)
## Save workbook
saveWorkbook(wb, "_data/test.xlsx", overwrite = TRUE)

print(paste("% Agreement between RF & NN", 
            (100 * sum(predictions[,2] == predictions[, 3])/nrow(predictions)) %>%
              round(1), "%"))