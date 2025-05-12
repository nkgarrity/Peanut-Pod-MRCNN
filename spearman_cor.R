cor_test <- read.csv("./Img_roll_correlation.csv", header = TRUE)

resJ <- cor.test(cor_test$Jumbo_Img, cor_test$Jumbo_Roll, method = "spearman")
resFarm <- cor.test(cor_test$farmer_fancy_img, cor_test$farmer_fancy_roll, method = "spearman")
resN <- cor.test(cor_test$No1_Img, cor_test$No1_Roll, method = "spearman")

resJ
resFarm
resN