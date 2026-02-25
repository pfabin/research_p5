library(tidyverse)

barcelona <- read.csv("/Users/petarfabinger/Desktop/University/AA:: BA/Y3S2P4/Research/data/barcelona.csv")
london <- read.csv("/Users/petarfabinger/Desktop/University/AA:: BA/Y3S2P4/Research/data/london.csv")
madrid <- read.csv("/Users/petarfabinger/Desktop/University/AA:: BA/Y3S2P4/Research/data/madrid.csv")
delhi <- read.csv("/Users/petarfabinger/Desktop/University/AA:: BA/Y3S2P4/Research/data/delhi.csv")
nyc <- read.csv("/Users/petarfabinger/Desktop/University/AA:: BA/Y3S2P4/Research/data/nyc.csv")
paris <- read.csv("/Users/petarfabinger/Desktop/University/AA:: BA/Y3S2P4/Research/data/paris.csv")

madrid$X <- as.character(madrid$X)
madrid$parse_count <- as.character(madrid$parse_count)
madrid$rating_review <- as.character(madrid$rating_review)

full <- bind_rows(barcelona, delhi, london, madrid, nyc, paris)

set.seed(2)
sample <- full %>% 
  group_by(city) %>%
  sample_frac(0.10) %>%
  ungroup()

sample[sample == ""] <- NA
sum(is.na(sample)) 
# 24 NA's, due to file corruption, tuple overlap

sample <- na.omit(sample)

# checking if aligned
sample$parse_count <- as.numeric(sample$parse_count) 

# Removing unnecessary columns which are factless
sample <- sample[,-c(1,2,6,8,12)]

# Cleaning data domains
sample$rating_review <- as.numeric(sample$rating_review)
sample$sample <- ifelse(sample$sample == "Positive", 1,0)
sample$bin <- ifelse(sample$rating_review %in% c(4,5), "positive", 
                            ifelse(sample$rating_review %in% c(1,2), "negative", "neutral"))
sample$date <- as.Date(sample$date, format = "%B %d, %Y")

# Anomalies with the names, some are not fully developed, so change is done
unique(sample$city) 
sample$city[startsWith(sample$city, "Barcelona")] <- "barcelona"
sample$city[startsWith(sample$city, "London")] <- "london"
sample$city[startsWith(sample$city, "New_Delhi")] <- "delhi"
sample$city[startsWith(sample$city, "Madrid")] <- "madrid"
sample$city[startsWith(sample$city, "New_York")] <- "nyc"
sample$city[startsWith(sample$city, "Paris")] <- "paris"
unique(sample$city) # double check 

# Re arranged for simplicity (GPT generated)
sample <- sample %>%
  relocate(review_full, .after = last_col())
sample <- sample %>%
  relocate(bin, .after = rating_review)

write.csv(sample, file = "/Users/petarfabinger/Desktop/University/AA:: BA/Y3S2P4/Research/data/clean.csv")