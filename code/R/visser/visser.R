

# visser

library(tidyverse)
library(depmixS4)
library(hmmr)
library(fpp3)
library(TTR)

# -------------------------------------------------------------------------

speed1 %>% 
  ggplot() +
  geom_density(aes(x = Pacc)) 

speed1 %>% 
  ggplot() +
  geom_density(aes(x = RT)) 

data("sp500")
sp500 

data("disc42")
nti = attr(disc42, "ntimes")
nti

data("discrimination")
discrimination



