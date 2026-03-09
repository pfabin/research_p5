library(tidytext)
library(tidyverse)
library(stringr)
library(SnowballC)
library(quanteda)
library(tm)
library(e1071)
library(quanteda)
library(quanteda.textmodels)
library(caret)
data <- read.csv("/Users/petarfabinger/Desktop/University/AA:: BA/Y3S2P4/Research/data/clean.csv")
data$date <- as.Date(data$date, format = "%B %d, %Y")
data <- data[, -1]
data$doc_id <- 1:nrow(data)
data <- data %>%
  relocate(doc_id, .before = title_review)
### All this code sets the data up after cleaning

# Corpus
corp <- corpus(data, text_field = "review_full")
docvars(corp, "bin") <- data$bin
### This creates a corpus for reviews, bin is the class sentiment

# BOW Model
toks <- tokens(
  corp,
  remove_punct = TRUE,
  remove_symbols = TRUE,
  remove_separators = TRUE,
  remove_url = TRUE
) %>%
  tokens_tolower()
toks <- tokens_remove(toks, pattern = stop_words$word)
dfm_all <- dfm(toks)
dfm_all <- dfm_trim(dfm_all, min_docfreq = 5)
### Tokens are cleaned and lowered, also stop words removed, 
### created into a  matrix with only words that are seen at least 5 times

# Model
set.seed(2)
train_id <- sample(seq_len(ndoc(dfm_all)), size = 0.8 * ndoc(dfm_all))
dfm_train <- dfm_all[train_id, ]
dfm_test  <- dfm_all[-train_id, ]
y_train <- data$bin[train_id]
y_test  <- data$bin[-train_id]
### Everything split 80,20 into train test and then preformed model

nb_model <- textmodel_nb(dfm_train, y = y_train)
pred <- predict(nb_model, newdata = dfm_test)
conf <- confusionMatrix(as.factor(pred), as.factor(y_test))
conf
f1_scores <- conf$byClass[,"F1"]
f_macro <- mean(f1_scores)
f_macro
### Recall (How many did we correctly discover for class c_i) is 0.73 negative, 0.86 positive, and 0.57 neutral
### Due to the imbalance the neutral class also being hard to predict, these are the scores.
### precision = 0.62, 0.97, 0.33

### F1 scores are 0.68 negative, 0.91 positive, and 0.42 neutral, again it is hard to catch neutral

### Macro treats all classes the same, dividing f1 by |c|, value is 0.667, mainly dropped from the neutral class.
### 0.67 is a solid baseline, but the model works extremely well with negative and positive reviews,
### It is mainly the neutral review which causes issues.

# TF-IDF 
### Applying the tf-idf calculation
dfm_train_tfidf <- dfm_tfidf(dfm_train)
dfm_test_tfidf  <- dfm_tfidf(dfm_test)

nb_model_tfidf <- textmodel_nb(dfm_train_tfidf, y = y_train)
pred_tfidf <- predict(nb_model_tfidf, newdata = dfm_test_tfidf)
conf_tfidf <- confusionMatrix(as.factor(pred_tfidf), as.factor(y_test))
conf_tfidf
### recall = 0.71 neg, 0.85 positive, 0.53 neutral. The recall has dropped slightly.
### precision = 0.57, 0.96, 0.30

f1_tfidf <- conf_tfidf$byClass[, "F1"]
f_macro_tfidf <- mean(f1_tfidf, na.rm = TRUE)
f_macro_tfidf

### The model has a lower f1 macro, and also a lower recall, this is not good.
### This could be from the vocab min setter, but otherwise it causes explosion
### perhaps setting limit to 2 instead of 5?

# Negation BoW model
negation_words <- c("not", "no", "never", "n't")
### adding the negated lexicon, maybe this should be expanded
custom_stopwords <- stop_words$word[!stop_words$word %in% negation_words]
toks_neg <- tokens(
  corp,
  remove_punct = TRUE,
  remove_symbols = TRUE,
  remove_separators = TRUE,
  remove_url = TRUE
) %>%
  tokens_tolower()
toks_neg <- tokens_remove(toks_neg, pattern = custom_stopwords)

toks_train_neg <- toks_neg[train_id]
toks_test_neg  <- toks_neg[-train_id]
dfm_train_neg <- dfm(toks_train_neg)
dfm_train_neg <- dfm_trim(dfm_train_neg, min_docfreq = 5)
dfm_test_neg <- dfm(toks_test_neg)
dfm_test_neg <- dfm_match(dfm_test_neg, features = featnames(dfm_train_neg))

nb_model_neg <- textmodel_nb(dfm_train_neg, y = y_train)
pred_neg <- predict(nb_model_neg, newdata = dfm_test_neg)
conf_neg <- confusionMatrix(as.factor(pred_neg), as.factor(y_test))
conf_neg
### recall = 0.74 negative, 0.87 positive, 0.59 neutral
### precision = 0.63, 0.97, 0.33
f1_neg <- conf_neg$byClass[, "F1"]
f_macro_neg <- mean(f1_neg, na.rm = TRUE)
f_macro_neg
### A little more successful, with 0.673 on the f1 macro, the numbers on recall and precision 
### are just slightly higher. This shows that negation indeed affects the model

# Comparison
f_macro %>% round(3)
f_macro_tfidf %>% round(3)
f_macro_neg %>% round(3)

