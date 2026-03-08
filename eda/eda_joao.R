library(tidyverse)
install.packages("ggwordcloud")
library(ggwordcloud)

data <- read.csv("~/OneDrive - Colégio Atlântico/Maastricht/Y3/Research Project/Project/Clean_Sample_Data.csv")
data$date <- as.Date(data$date)
data <- data[,-1]

# Proportions per city
city_sentiment_gap <- data %>%
  group_by(city, bin) %>%
  summarise(count = n()) %>%
  mutate(proportion = count / sum(count))

ggplot(city_sentiment_gap, aes(x = city, y = proportion, fill = bin)) +
  geom_bar(stat = "identity", position = "stack") +
  scale_fill_manual(values = c("negative" = "#e41a1c", "neutral" = "#377eb8", "positive" = "#4daf4a")) +
  labs(title = "Sentiment Proportion by City", 
       subtitle = "Comparing customer satisfaction across 6 global cities",
       x = "City", y = "Proportion of Reviews") +
  theme_minimal()

# We can see that Parisian people tend to be harsher when reviewing restaurants
# followed by people in Barcelona


# Aggregate average rating by month/year
time_trend <- data %>%
  mutate(month = floor_date(date, "month")) %>%
  group_by(month, city) %>%
  summarise(avg_rating = mean(rating_review, na.rm = TRUE), .groups = "drop")

# Plot the trend
ggplot(time_trend, aes(x = month, y = avg_rating, color = city)) +
  geom_line(size = 1) +
  geom_smooth(method = "loess", se = FALSE, linetype = "dashed", size = 0.5) + # Adds a trend line
  labs(title = "Sentiment Evolution (2010-2026)",
       subtitle = "Average monthly ratings per city",
       x = "Year", y = "Average Rating") +
  theme_minimal()

ggplot(time_trend, aes(x = month, y = avg_rating, color = city)) +
  geom_line(alpha = 0.5) + 
  geom_smooth(method = "loess", se = FALSE, color = "black", size = 0.8) + 
  facet_wrap(~ city, scales = "free_y") +  # This creates the separate plots
  theme_minimal() +
  theme(legend.position = "none") +
  labs(
    title = "Customer Perception Trends by City",
    subtitle = "Monthly average ratings with trend lines",
    x = "Date",
    y = "Average Rating"
  )
# We can see that around 2008 the financial crisis not only affected everyone's finances
# but also how people reacted to the overall restaurant industry. Maybe we could look
# through what were the most common topics argued in that time period.

target_city <- "delhi"

time_trend %>%
  filter(city == target_city) %>%
  ggplot(aes(x = month, y = avg_rating)) +
  geom_line(color = "steelblue", size = 1) + 
  geom_smooth(method = "loess", se = FALSE, color = "black", fill = "lightgrey", size = 0.8) + 
  theme_minimal() +
  labs(
    title = "Customer Perception Trend:", str_to_title(target_city),
    subtitle = "Monthly average ratings with trend lines",
    x = "Date",
    y = "Average Rating"
  )


# Check the top 50 most common words on negative reviews
neg_words <- tokens_clean %>%
  inner_join(data %>% filter(bin == "negative") %>% select(doc_id), by = "doc_id") %>%
  count(word, sort = TRUE) %>%
  top_n(50)

# Generate the cloud
ggplot(neg_words, aes(label = word, size = n, color = n)) +
  geom_text_wordcloud() +
  scale_size_area(max_size = 20) +
  scale_color_gradient(low = "darkred", high = "red") +
  theme_minimal() +
  labs(title = "Top 50 Keywords in Negative Reviews")

View(neg_words)

