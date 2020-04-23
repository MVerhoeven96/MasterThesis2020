## Load data -------------------------------------------------------------------
Phone_mood_data <- read.csv("Final.csv", stringsAsFactors = FALSE)
Phone_mood_data <- transform(Phone_mood_data, date = as.Date(date))

## Application per mood split ----------------------------------------------------
FH <- Phone_mood_data %>%
  filter(date <= as.Date("2019-03-09"))
LH <- Phone_mood_data %>%
  filter(date >= as.Date("2019-03-10"))
F3D <- Phone_mood_data %>%
  filter(date <= as.Date("2019-02-23"))
L3D <- Phone_mood_data %>%
  filter(date >= as.Date("2019-03-24"))

## Mood per application FH -----------------------------------------------------
ggplot(data = FH, aes(x= mood_day, y = category, fill = category)) +
  geom_bar(stat = "identity") +
  ggtitle("Model FH") +
  xlab("Mood") + ylab("Application category") +
  labs(fill = "Application category") +
  scale_fill_manual(values = c("#2166AC","lightgrey", "#D1E5F0", "#92C5DE", "#4393C3")) +
  theme(axis.text.y = element_blank(),
        panel.background = element_rect(fill = "white", colour = "white"),
        plot.background = element_rect(fill = "white", colour = "white"))

## Mood per application LH -----------------------------------------------------
ggplot(data = LH, aes(x= mood_day, y = category, fill = category)) +
  geom_bar(stat = "identity") +
  ggtitle("Model LH") +
  xlab("Mood") + ylab("Application category") +
  labs(fill = "Application category") +
  scale_fill_manual(values = c("#2166AC","lightgrey", "#D1E5F0", "#92C5DE", "#4393C3")) +
  theme(axis.text.y = element_blank(),
        panel.background = element_rect(fill = "white", colour = "white"),
        plot.background = element_rect(fill = "white", colour = "white"))

## Mood per application F3D ----------------------------------------------------
ggplot(data = F3D, aes(x= mood_day, y = category, fill = category)) +
  geom_bar(stat = "identity") +
  ggtitle("Model F3D") +
  xlab("Mood") + ylab("Application category") +
  labs(fill = "Application category") +
  scale_fill_manual(values = c("#2166AC","lightgrey", "#D1E5F0", "#92C5DE", "#4393C3")) +
  theme(axis.text.y = element_blank(),
        panel.background = element_rect(fill = "white", colour = "white"),
        plot.background = element_rect(fill = "white", colour = "white"))

## Mood per application L3D ----------------------------------------------------
ggplot(data = L3D, aes(x= mood_day, y = category, fill = category)) +
  geom_bar(stat = "identity") +
  ggtitle("Model L3D") +
  xlab("Mood") + ylab("Application category") +
  labs(fill = "Application category") +
  scale_fill_manual(values = c("#2166AC","lightgrey", "#D1E5F0", "#92C5DE", "#4393C3")) +
  theme(axis.text.y = element_blank(),
        panel.background = element_rect(fill = "white", colour = "white"),
        plot.background = element_rect(fill = "white", colour = "white"))

## Mood per application AM -----------------------------------------------------
ggplot(data = Phone_mood_data, aes(x= mood_day, y = category, fill = category)) +
  geom_bar(stat = "identity") +
  ggtitle("Model AM") +
  xlab("Mood") + ylab("Application category") +
  labs(fill = "Application category") +
  scale_fill_manual(values = c("#2166AC","lightgrey", "#D1E5F0", "#92C5DE", "#4393C3")) +
  theme(axis.text.y = element_blank(),
        panel.background = element_rect(fill = "white", colour = "white"),
        plot.background = element_rect(fill = "white", colour = "white"))

## Pie Chart App ----------------------------------------------------------------
Categories_data <- c("Browser", "Entertainment", "Games", "Social Networking", "Utility")
Categories_count <- c(13816, 52566, 1728, 265575, 65437)
table <- data.frame(Categories_data, Categories_count)
colnames(table) <- c("Category", "Count")

pie(table[,2], labels = table[,1],
    col = c("white",  "#2166AC", "#D1E5F0", "#92C5DE", "#4393C3"),
    main = "Application categories")

## Pie Chart mood --------------------------------------------------------------
table(Phone_mood_data$mood_day, useNA = "always")
Mood_data <- c("Pleasant activation", "Pleasant deactivation", "Unpleasant activation", "Unpleasant deactivation")
Mood_count <- c(97544, 197890, 22844, 79630)
table <- data.frame(Mood_data, Mood_count)
colnames(table) <- c("Mood", "Count")

pie(table[,2], labels = table[,1],
    col = c("#2166AC", "#D1E5F0", "#92C5DE", "#4393C3"),
    main = "Daily mood")
