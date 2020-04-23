## Load data -------------------------------------------------------------------
phone <- read.csv("phone_constructed.csv", stringsAsFactors = FALSE)
mood <- read.csv("mood_constructed.csv", stringsAsFactors = FALSE)

## Merge all variables ---------------------------------------------------------
final <- merge(phone, mood, by = c("date", "user_id", "day_part"))
final <- final %>%
  select(-X.x, -X.y, -Y_activeness, -X_pleasure, -duration)

## Splits for models -----------------------------------------------------------
first_half <- final %>%
  filter(date <= as.Date("2019-03-09"))
second_half <- final %>%
  filter(date >= as.Date("2019-03-10"))
first3_day <- final %>%
  filter(date <= as.Date("2019-02-23"))
last3_day <- final %>%
  filter(date >= as.Date("2019-03-24"))
