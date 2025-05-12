library(tidyverse)
library(lme4)
library(lmerTest)
library(knitr)
library(kableExtra)


pa_in <- read_csv("./pod_traits.csv")

pheno_cols <- c("Year", "Location", "NC_Accession","pod_area","pod_width", "Block")
pa_in <- pa_in[, pheno_cols]
pa_in$Block <- as.numeric(pa_in$Block)


gapit_names <- read.csv("./gapit_names.csv")
gapit_names$NC_Accession <- gapit_names$taxa

merged_data <- merge(pa_in, gapit_names, by = "NC_Accession", all.x = TRUE)
merged_data$NC_Accession <- merged_data$taxa.1
merged_data <- merged_data[,1:6]
merged_data$Env <- paste(merged_data$Year, merged_data$Location, sep = "_")


envmodel_width <- lmer(pod_width ~ (1 | Env/Block) + (1 | NC_Accession) + (1 | NC_Accession:Env), data = merged_data)

envmodel_area <- lmer(pod_area ~ (1 | Env/Block) + (1 | NC_Accession) + (1 | NC_Accession:Env), data = merged_data)

#variance components
vc_width <- VarCorr(envmodel_width)
vc_area <- VarCorr(envmodel_area)

#AREA
VG_area <- unlist(vc_area$NC_Accession)
VGE_area <- unlist(vc_area$`NC_Accession:Env`)
Verr_area <- (sigma(envmodel_area)^2)
H2_area <- VG_area / (VG_area + VGE_area + Verr_area)


rea_area <- ranova(envmodel_area)
coeffs_area <- coef(envmodel_area)
ran_eff_area <- ranef(envmodel_area)

#WIDTH
VG_width <- unlist(vc_width$NC_Accession)
VGE_width <- unlist(vc_width$`NC_Accession:Env`)
Verr_width <- (sigma(envmodel_width)^2)
H2_width <- VG_width / (VG_width + VGE_width + Verr_width)


rea_width <- ranova(envmodel_width)
coeffs_width <- coef(envmodel_width)
ran_eff_width <- ranef(envmodel_width)

area_df <- as.data.frame(rea_area)
width_df <- as.data.frame(rea_width)

kable(area_df, format = "html", caption = "Pod Area Anova Results", digits = 2000) %>%
  kable_styling(full_width = FALSE)

kable(width_df, format = "html", caption = "Pod Width Anova Results", digits = 2000) %>%
  kable_styling(full_width = FALSE)

m_area <- mean(merged_data$pod_area)
s_area <- sd(merged_data$pod_area)

distro_area <- ggplot(data = merged_data, aes(x = pod_area)) +
  geom_histogram(binwidth = 5, fill = "red", color = "black", alpha = 0.7) +
  labs(title = "Mean Pod Area by Plot", x = "Mean Pod Area (mm^2)", y = "Frequency") +
  geom_text(aes(x = m_area - 110, y= 75, label = paste("Mean =", round(m_area, 2))), vjust = -0.5, color = "black") +
  geom_text(aes(x = m_area - 50 , y = 75, label = paste("SD =", round(s_area, 2))), vjust = -0.5, color = "black") +
  geom_text(aes(x = m_area - 200, y = 100, label = "Bin Size = 5mm^2"), vjust = -0.5, color = "black")



distro_area +
  stat_function(fun = function(x) dnorm(x, mean = mean(merged_data$pod_area), sd = sd(merged_data$pod_area)) * length(merged_data$pod_area) * 5, color = "blue", size = 1)


m_width <- mean(merged_data$pod_width)
s_width <- sd(merged_data$pod_width)


distro_width <- ggplot(data = merged_data, aes(x = pod_width)) +
  geom_histogram(binwidth = .1, fill = "red", color = "black", alpha = 0.7) +
  labs(title = "Mean Pod Width by Plot", x = "Mean Pod Width (mm)", y = "Frequency") +
  geom_text(aes(x = m_width - 2, y= 75, label = paste("Mean =", round(m_width, 2))), vjust = -0.5, color = "black") +
  geom_text(aes(x = m_width - 1 , y = 75, label = paste("SD =", round(s_width, 2))), vjust = -0.5, color = "black") +
  geom_text(aes(x = m_width - 4, y = 115, label = "Bin Size = 0.1mm"), vjust = -0.5, color = "black")


distro_width +
  stat_function(fun = function(x) dnorm(x, mean = mean(merged_data$pod_width), sd = sd(merged_data$pod_width)) * length(merged_data$pod_width) * .1, color = "blue", size = 1)

