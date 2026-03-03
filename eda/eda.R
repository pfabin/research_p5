library(tidyverse)
data <- read.csv("/Users/petarfabinger/Desktop/University/AA:: BA/Y3S2P4/Research/data/clean.csv")
data$date <- as.Date(data$date, format = "%B %d, %Y")
data <- data[,-1]

# EDA for bins
ggplot(data, aes(x = bin, fill = bin)) +
  geom_bar() +
  labs(
    title = "Distribution of Rating Categories",
    x = "Rating Category",
    y = "Count",
    fill = "Rating Category"
  ) +
  theme_minimal() +
  scale_fill_manual(values = c("red", "yellow", "green"))

# Review length
data$review_length <- nchar(data$review_full)
head(data$review_length)

## Visualise the distribution
ggplot(data, aes(x = review_length)) +
  geom_density(fill = "purple") +
  labs(x = "Review Length", y = "Density",
  title = "Review Length Density Plot")
### Extremely skewed, we will apply winsorisation at q4 cap

summary(data$review_length)
### median is 324, this is the average review. The max characters is 16805, and minimum is 4.
### There are several outliers.

data$wins_review_length <- pmin(data$review_length, 542)
ggplot(data, aes(x = wins_review_length)) +
  geom_density(fill = "purple", alpha = 0.4) +
  labs(x = "Review Length", y = "Density",
       title = "Review Length Winsorised Density Plot")

correlation <- cor(data$review_length, data$rating_review)
### -0.176 is a weak negative relationship, as review length increases, review rating decreases
### this could be due to negative reviews being longer or completely short, causing the relationship to be weak.

data$log_review_length <- log(data$review_length)
ggplot(data, aes(x = bin, y = review_length, fill = bin)) +
  geom_boxplot() +
  scale_y_continuous(trans = "log10") +
  labs(title = "Review Length Distribution by Category", x = "Category", y = "Review Length") +
  theme_minimal()

### Negative is the widest IQR, this means they are not so consistent, however it has the highest median character count, 
### so the negative reviews tend to be longer. Positive has heavy outlier that spread larger than the other two, but
### the review length median is shortest. Neutral is in the middle, more like positive in IQR size with one big outlier.
### This could be that neutral gives more evaluation

# ANOVA
### H0: There is no difference in review length between the sentiment categories.
### H1: There is a significant difference in review length between at least two sentiment categories.

anova_result <- aov(log_review_length ~ bin, data = data)
summary(anova_result)
## reject H0, there is a significant difference

tukey_result <- TukeyHSD(anova_result)
tukey_result
### Tukey's says all are different between eachother statistically, there is class imbalance, even on log level.

# Returning customers
## Filtering by author with more than 1 review
customer_review_count <- data %>%
  group_by(author_id) %>%
  summarise(review_count = n())
returning_customers <- customer_review_count %>%
  filter(review_count > 1)
(nrow(returning_customers)/nrow(data)) %>% round(3)
## 20.8% of the data is repeat users

summary(returning_customers$review_count)
nrow(returning_customers[returning_customers$review_count >= 3,])
## Interesting that the median and q1 is 2, meaning most users write a review
## and then only one more review before quiting. There are several reviewers 
## past the q3 boundary, namely 22586, 

review_freq <- customer_review_count %>%
  count(review_count) %>%
  mutate(prop = n / sum(n))
review_freq <- review_freq %>%
  filter(review_count > 1)

ggplot(review_freq, aes(x = review_count, y = prop)) +
  geom_line(linewidth = 0.6) + 
  geom_area(fill = "lightblue", alpha = 0.4) +
  labs(
    title = "Review Count Per Reviewer Density Plot (log transformed)",
    x = "Review Count",
    y = "Proportion") +
  scale_x_log10() +
  theme_minimal()

