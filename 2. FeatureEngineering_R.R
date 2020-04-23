
## Adding date variables --------------------------------------------------------
mood_data$date <- as.IDate(mood_data$sent_time)
mood_data$time <- as.ITime(mood_data$sent_time)

mood_data <- mood_data %>%
  filter(date >= "2019-02-21" & date <= "2019-03-26")

phone_data$startTime <- ymd_hms(phone_data$startTime)
phone_data$date <- as.IDate(phone_data$startTime)
phone_data$time <- as.ITime(phone_data$startTime)

## Merging categories ---------------------------------------------------------
phone_data <- phone_data %>%
  mutate(category = ifelse(category == "Auto & Vehicles"  | category == "Art & Design" 
                           | category == "Books & Reference" | category == "Entertainment" 
                           | category == "Food & Drink" | category == "Music & Audio"
                           | category == "News & Magazines" | category == "Sports" | category == "Video Players & Editors"
                           | category == "Others" | category == "Health & Fitness" , "Entertainment", category))%>%
  mutate(category = ifelse(category == "Background Process"  | category == "Maps & Navigation" 
                           | category == "Finance" | category == "House & Home"
                           | category == "Medical" | category == "Personalization" 
                           | category == "Photography" | category == "Shopping"| category == "Tools"
                           | category == "Travel & Local" | category == "Weather" | category == "Word"
                           , "Utility", category))%>%
  mutate(category = ifelse(category == "Communication"  | category == "Dating" 
                           | category == "Social" | category == "Lifestyle" 
                           , "Social Networking", category))%>%
  mutate(category = ifelse(category == "Action"  | category == "Adventure" | category == "Board" 
                           | category == "Arcade" | category == "Card" | category == "Casino"
                           | category == "Casual" | category == "Strategy" | category == "Trivia"
                           | category == "Puzzle" | category == "Racing"
                           | category == "Racing, Action & Adventure" | category == "Simulation"
                           , "Games", category))%>%
  mutate(category = ifelse(category == "Business"  | category == "Education" 
                           | category == "Productivity", "Browser", category))

## Mood to Circumplex model-----------------------------------------------------
mood_data <- mood_data %>%
  mutate(Y_activeness = (anxious*0.9 + bored*-0.69 + gloomy*-0.5 + calm*-0.69 + stressed*0.867 + content*-0.69 + cheerful*0.69 + tired*-0.9 + energetic*0.867 + upset*0) / 6.794) %>%
  mutate(X_pleasure = (anxious*-0.1 + bored*-0.69 + gloomy*-0.867 + calm*0.69 + stressed*-0.5 + content*0.69 + cheerful*0.69 + tired*-0.1 + energetic*0.5 + upset*-1) / 5.827)

## Feature: Duration -----------------------------------------------------------
phone_data$startTime <- ymd_hms(phone_data$startTime)
phone_data$startTime <- as.ITime(phone_data$startTime)

phone_data <- phone_data %>%
  mutate(endTimeMillis = ifelse(endTimeMillis < startTimeMillis, 1.550876e+12 + endTimeMillis, endTimeMillis))

phone_data$duration <- phone_data$endTimeMillis - phone_data$startTimeMillis

## Phone features: day_part, counts, durations, notif.--------------------------
phone <- phone_data
phone <- transform(phone, time = as.character(time))
phone$time_sec <- unlist(lapply(lapply(strsplit(phone$time, ":"), as.numeric), function(x) x[1]*60^2+x[3]))
phone$hours <- round(phone$time_sec/3600, digits = 1)

phone_copy <- phone %>%
  mutate(day_part = ifelse(hours > 5 & hours < 12, "morning", ifelse(hours > 11 & hours < 18, "afternoon", ifelse(hours > 17 & hours <= 23, "evening", NA_integer_))))%>%
  mutate(game_count = ifelse(category == "Games", 1, 0)) %>%
  mutate(game_dur = ifelse(category == "Games", phone$duration, 0)) %>%
  mutate(util_count = ifelse(category == "Utility", 1, 0)) %>%
  mutate(util_dur = ifelse(category == "Utility", phone$duration, 0)) %>%
  mutate(entert_count = ifelse(category == "Entertainment", 1, 0)) %>%
  mutate(entert_dur = ifelse(category == "Entertainment", phone$duration, 0)) %>%
  mutate(social_count = ifelse(category == "Social Networking", 1, 0)) %>%
  mutate(social_dur = ifelse(category == "Social Networking", phone$duration, 0)) %>%
  mutate(brow_count = ifelse(category == "Browser", 1, 0)) %>%
  mutate(brow_dur = ifelse(category == "Browser", phone$duration, 0)) %>%
  mutate(notif_count = ifelse(notification == "True", 1, 0))

phone_copy <- phone_copy %>%
  mutate_at(vars(game_count:notif_count), ~replace(., is.na(.), 0)) %>%
  na.omit(cols = c("day_part"))

phone_grouped <- phone_copy %>%
  group_by(user_id, date, day_part) %>%
  summarise(min_dur = min(duration), max_dur = max(duration),game_count = sum(game_count), game_dur = sum(game_dur), util_count = sum(util_count), util_dur = sum(util_dur), entert_count = sum(entert_count), entert_dur = sum(entert_dur),
            social_count = sum(social_count), social_dur = sum(social_dur), brow_count = sum(brow_count), brow_dur = sum(brow_dur), duration = sum(duration), notif_count = sum(notif_count)) 
phone_grouped$total_count <- (phone_grouped$game_count + phone_grouped$util_count + phone_grouped$entert_count + phone_grouped$social_count + phone_grouped$brow_count)
phone_grouped$total_dur <- (phone_grouped$game_dur + phone_grouped$util_dur + phone_grouped$entert_dur + phone_grouped$social_dur + phone_grouped$brow_dur)

write.csv(phone_grouped, "phone_constructed.csv")

#Mood features: day_part, mood--------------------------------------------------
mood<- mood_data
mood <- transform(mood, time = as.character(time))
mood$time_sec <- unlist(lapply(lapply(strsplit(mood$time, ":"), as.numeric), function(x) x[1]*60^2+x[3]))
mood$hours <- round(mood$time_sec/3600, digits = 1)

mood_copy <- mood %>%
  mutate(day_part = ifelse(hours > 5 & hours < 12, "morning", ifelse(hours > 11 & hours < 18, "afternoon", ifelse(hours > 17 & hours <= 23, "evening", NA_integer_))))

mood_grouped <- mood_copy %>%
  select("user_id", "date", "day_part", "Y_activeness", "X_pleasure") %>%
  group_by(user_id, date, day_part) %>%
  summarise_all(mean) %>%
  mutate(X_pleasure = ifelse(X_pleasure < 0, "Unpleasant", ifelse(Y_activeness == 0, "Neutral", "Pleasant"))) %>%
  mutate(Y_activeness = ifelse(Y_activeness < 0, "Deactivation", ifelse(Y_activeness == 0, "Neutral", "Activation")))
mood_grouped$mood <- paste0(mood_grouped$X_pleasure, " ", mood_grouped$Y_activeness)

mood_grouped <- mood_grouped %>%
  mutate(mood = ifelse(mood == "NA NA", NA_character_, mood)) %>%
  mutate(mood = ifelse(mood == "Neutral Neutral", NA_character_, mood)) %>%
  mutate(mood = ifelse(mood == "Unpleasant Neutral", NA_character_, mood)) 

write.csv(mood_grouped, "mood_constructed.csv")

