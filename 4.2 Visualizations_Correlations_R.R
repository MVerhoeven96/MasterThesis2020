## Load data -------------------------------------------------------------------
final <- read.csv("Cleaned_datasets/final.csv", stringsAsFactors = FALSE)
final <- final %>%
  select(-date, -user_id, -day_part, -mood, -X)

## Correlation matrix ----------------------------------------------------------
final = as.matrix(final)
final_cor = round(cor(final),3)
corrplot(final_cor, method="number", type="upper", tl.col="black", tl.srt=45)

col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(final_cor, method="color", col=col(200), tl.cex = 1,
         type="upper", order="hclust", addCoef.col = "black", 
         tl.col="black", tl.srt = 45, sig.level = 0.01, insig = "blank", diag=FALSE)
