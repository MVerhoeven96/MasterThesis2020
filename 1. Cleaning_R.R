## Name: Maartje Verhoeven
## Studentnumber: u1273860

## Load packages ---------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(tidyverse)
library(e1071)
library(data.table)
library(lubridate)
library(chron)

## Load packages ---------------------------------------------------------------
setWD()

## Load data -------------------------------------------------------------------
mood_complete_dataset <- read.csv("mood_sampling_data.csv",
                                  stringsAsFactors = FALSE)
app_complete_dataset <- read.csv("mapp_categories.csv",
                                 stringsAsFactors = FALSE)
phone_complete_dataset <- read.csv("phone_use_data.csv",
                                   stringsAsFactors = FALSE)

## merge dataframes -------------------------------------------------------------------
clean_app_complete_dataset <- app_complete_dataset %>%
  select(app_id, name, category) %>%
  rename(application = app_id) 

phone_data <- merge(clean_app_complete_dataset, phone_complete_dataset, by = "application") %>%
  select(-application, -battery) %>%
  mutate_all(na_if,"")

mood_data <- mood_complete_dataset %>%
  select(-response_time, -duration, -enjoy, -activities_study, -activities_class, -activities_work,
         -activities_social, -activities_sports, - activities_commute, -activities_sleep, -activities_chore,
         -social_private, -social_public, - -social_partener, -social_friend, -social_family, -social_coworkers,
         -day_time_window, -activities_leisure, -social_partener) %>%
  mutate_all(na_if,"")

## Removing NA rows -----------------------------------------------------------
mood_data <- mood_data[!(rowSums(is.na(mood_data)) == 14),] 

## Exploring classes  ----------------------------------------------------------
class(mood_data$sent_time)
class(mood_data$anxious)
class(mood_data$bored)
class(mood_data$gloomy)
class(mood_data$calm)
class(mood_data$stressed)
class(mood_data$content)
class(mood_data$cheerful)
class(mood_data$tired)
class(mood_data$energetic)
class(mood_data$upset)
class(mood_data$envious)
class(mood_data$inferior)
class(mood_data$activity)
class(mood_data$social)
class(mood_data$date)

# Creating numeric and factor values
mood_data <- transform(mood_data, cheerful = as.integer(cheerful),
                       tired = as.integer(tired), energetic = as.integer(energetic))

class(phone_data$category)
class(phone_data$startTime)
class(phone_data$notification)
class(phone_data$session)
class(phone_data$user_id)
class(phone_data$date)

## Checking categorical labels--------------------------------------------------
unique(mood_data$social)
unique(mood_data$activity)
unique(mood_data$anxious)
unique(mood_data$user_id) #136

unique(phone_data$category) #44 different values
unique(phone_data$notification) #true/false
unique(phone_data$user_id) #124

unique(app_complete_dataset$app_id) #
unique(app_complete_dataset$category) #59

## Creating NA and removing outliers -------------------------------------------
mood_data <- mood_data %>%
  mutate(tired = ifelse(tired > 5, NA_integer_, tired)) %>%
  mutate(anxious = ifelse(anxious > 5, NA_integer_, anxious)) %>%
  mutate(bored = ifelse(bored > 5, NA_integer_, bored)) %>%
  mutate(gloomy = ifelse(gloomy > 5, NA_integer_, gloomy)) %>%
  mutate(calm = ifelse(calm > 5, NA_integer_, calm)) %>%
  mutate(stressed = ifelse(stressed > 5, NA_integer_, stressed)) %>%
  mutate(content = ifelse(content > 5, NA_integer_, content)) %>%
  mutate(cheerful = ifelse(cheerful > 5, NA_integer_, cheerful)) %>%
  mutate(energetic = ifelse(energetic > 5, NA_integer_, energetic)) %>%
  mutate(upset = ifelse(upset > 5, NA_integer_, upset)) %>%
  mutate(envious = ifelse(envious > 5, NA_integer_, envious)) %>%
  mutate(inferior = ifelse(inferior > 5, NA_integer_, inferior))

# Check NA values
which(is.na(mood_data$sent_time))
which(is.na(mood_data$anxious))
which(is.na(mood_data$bored))
which(is.na(mood_data$gloomy))
which(is.na(mood_data$calm))
which(is.na(mood_data$stressed))
which(is.na(mood_data$content))
which(is.na(mood_data$cheerful))
which(is.na(mood_data$tired))
which(is.na(mood_data$energitic))
which(is.na(mood_data$upset))
which(is.na(mood_data$envious))
which(is.na(mood_data$inferior))
which(is.na(mood_data$activity))
which(is.na(mood_data$social))
which(is.na(mood_data$date))
which(is.na(mood_data$time))

#NA in envious, inferior, activity, social around 600/9640 missing. 

which(is.na(phone_data$category))
which(is.na(phone_data$startTime))
which(is.na(phone_data$notification))
which(is.na(phone_data$session))
which(is.na(phone_data$user_id))
which(is.na(phone_data$date))

#NA in category around 1000/472659 

## Counts per category ----------------------------------------------------------
table(mood_data$tired, useNA = "always")
table(mood_data$anxious, useNA = "always")
table(mood_data$bored, useNA = "always")
table(mood_data$gloomy, useNA = "always")
table(mood_data$calm, useNA = "always")
table(mood_data$stressed, useNA = "always")
table(mood_data$content, useNA = "always")
table(mood_data$cheerful, useNA = "always")
table(mood_data$energetic, useNA = "always")
table(mood_data$upset, useNA = "always")
table(mood_data$envious, useNA = "always")
table(mood_data$inferior, useNA = "always")

table(phone_data$notification, useNA = "always")
table(phone_data$category, useNA = "always")



