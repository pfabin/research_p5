library(tidytext)
library(tidyverse)
library(stringr)
library(SnowballC)
library(quanteda)
data <- read.csv("/Users/petarfabinger/Desktop/University/AA:: BA/Y3S2P4/Research/data/clean.csv")
data$date <- as.Date(data$date, format = "%B %d, %Y")
data <- data[,-1]
data$doc_id <- 1:nrow(data)
data <- data %>%
  relocate(doc_id, .before = title_review)

# Tokenisation
negators <- c("not", "no", "never", "n't")
tokens <- data %>%
  select(doc_id, city, review_full) %>%
  unnest_tokens(word, review_full, to_lower = TRUE) %>%
  group_by(doc_id) %>%
  mutate(word_order = row_number()) %>%
  ungroup()

tokens <- tokens %>%
  group_by(doc_id) %>%
  mutate(next_word = lead(word)) %>%
  ungroup()

tokens <- tokens %>%
  group_by(doc_id) %>%
  mutate(next_word = lead(word)) %>%
  ungroup()

tokens <- tokens %>%
  group_by(doc_id) %>%
  mutate(remove_flag = lag(word %in% negators, default = FALSE)) %>%
  ungroup() %>%
  filter(!remove_flag)

# Stop word removal
data("stop_words")
tokens_clean <- tokens %>%
  anti_join(stop_words, by = "word")

# Stemming
tokens_clean <- tokens_clean %>%
  mutate(word = wordStem(word))

# NLP

## BoW
tokens_clean %>%
  count(city, word, sort = TRUE) %>%
  head(10)

top10_city <- tokens_clean %>%
  count(city, word) %>%
  group_by(city) %>%
  slice_max(order_by = n, n = 10, with_ties = FALSE) %>%  # exactly 10
  arrange(city, desc(n)) %>%
  ungroup()

## TF-IDF
vocab <- tokens_clean %>%
  distinct(doc_id, word) %>%
  count(word, name = "docfreq") %>%
  filter(docfreq >= 300)

tokens_small <- tokens_clean %>%
  semi_join(vocab, by = "word")

### corpus creation
corp <- corpus(data, text_field = "review_full")
docvars(corp, "city") <- data$city
toks <- tokens(
  corp,
  remove_punct = TRUE,
  remove_symbols = TRUE,
  remove_separators = TRUE,
  remove_url = TRUE
) |>
  tokens_tolower()
toks <- tokens_keep(toks, pattern = vocab$word)
dfm_all <- dfm(toks)
dfm_tfidf <- dfm_tfidf(dfm_all)
dfm_city <- dfm_group(dfm_all, groups = docvars(dfm_all, "city"))
dfm_city_tfidf <- dfm_tfidf(dfm_city)
top10_tfidf_by_city <- lapply(docnames(dfm_city_tfidf), function(ct) {
  x <- dfm_city_tfidf[ct, ]
  out <- data.frame(
    city = ct,
    word = names(topfeatures(x, 10)),
    tfidf = as.numeric(topfeatures(x, 10))
  )
  out
})

top10_tfidf_by_city <- do.call(rbind, top10_tfidf_by_city)
top10_tfidf_by_city