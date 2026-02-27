library(tidyverse)

barcelona <- read.csv("")
london <- read.csv("")
madrid <- read.csv("")
delhi <- read.csv("")
nyc <- read.csv("")
paris <- read.csv("")

# Joining transformation
madrid$X <- as.character(madrid$X)
madrid$parse_count <- as.character(madrid$parse_count)
madrid$rating_review <- as.character(madrid$rating_review)
full <- bind_rows(barcelona, delhi, london, madrid, nyc, paris)

# Duplicates
any(duplicated(full$review_id))
full <- full[!duplicated(full$review_id), ]
any(duplicated(full$review_id))
## 2756038 before removal of duplciation; 2756028 after removal, 10 duplicates

# NA's
full[full == ""] <- NA
sum(is.na(full)) 
## 114 Na's out of 2.75million, omitting won't hurt
## After inspection, Na's are caused from file corruption
full <- na.omit(full)

# Column cleaning
## Removing unnecessary columns which are irrelevant to us
sample <- sample[,-c(1,2,6,8,12)]
full <- full[, !names(full) %in% c("X", "parse_count", "review_id", "review_preview", "url_restaurant")]
## removed X, parse_count, and review_id due to its surrogate nature, no analytical quality
## removed review_preview due to title and full review present in data
## removed restaurant URL as it is redundant, could still use for more data...

# Cleaning data domains
full$rating_review <- as.numeric(full$rating_review)
full$sample <- ifelse(full$sample == "Positive", 1,0)
full$bin <- ifelse(full$rating_review %in% c(4,5), "positive", 
                     ifelse(full$rating_review %in% c(1,2), "negative", "neutral"))
full$date <- as.Date(full$date, format = "%B %d, %Y")

## Balance check
table(full$bin)

# Naming
## Anomalies with the names, some are not fully developed, so change is done
unique(full$city) 
full$city[startsWith(full$city, "Barcelona")] <- "barcelona"
full$city[startsWith(full$city, "London")] <- "london"
full$city[startsWith(full$city, "New_Delhi")] <- "delhi"
full$city[startsWith(full$city, "Madrid")] <- "madrid"
full$city[startsWith(full$city, "New_York")] <- "nyc"
full$city[startsWith(full$city, "Paris")] <- "paris"
unique(full$city) # double check 

# Re arranged for simplicity (this section is GPT generated)
full <- full %>%
  relocate(review_full, .after = last_col())
full <- full %>%
  relocate(bin, .after = rating_review)

# Sampling
set.seed(2)
p <- 0.10
sample <- full %>% 
  group_by(city) %>%
  sample_frac(p) %>%
  ungroup()

write.csv(sample, file = "")
