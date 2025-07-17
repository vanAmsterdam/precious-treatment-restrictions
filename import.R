library(haven)
library(data.table)
df_raw <- haven::read_sav(here::here("data", "PRECIOUS_treatmentrestrictions.sav"))
df_raw$timetodeath
df <- copy(df_raw)
setDT(df)
str(df_raw)

renames <- c(
"AGE_AGE"="age",
"SEX"="sex",
"DM"="dm",
"EARLYREST"="earlyrest",
"MINI05_NIHHS"="nihhs",
"DIAGN"="diagn",
"metoclopramide"="metoclopramide",
"paracetamol"="paracetamol",
"ceftriaxone"="ceftriaxone",
"ALIVE_FUP"="alive_fup",
"PRESTROKE_mRS"="prestroke_mrs",
"mrs_complete"="mrs_complete",
"country"="country"
)

oldnames <- names(renames)
newnames <- renames

setnames(df, oldnames, newnames)
df <- df[, .SD, .SDcols=newnames]

df[, prestrokedep:=prestroke_mrs>=3]

df[, table(prestroke_mrs, prestrokedep)]

df[, `:=`(
  age=as.numeric(age),
  sexmale=sex==1,
  dm=dm==1,
  earlyrest=as.logical(earlyrest),
  nihhs=as.numeric(nihhs),
  diagnstroke=diagn==1,
  deceased=alive_fup==1,
  mrsprestroke=as.numeric(prestroke_mrs),
  mrsendofstudy=as.numeric(mrs_complete)
)]

drop_vars <- c("sex", "diagn", "alive_fup", "mrs_complete", "prestroke_mrs")
keep_vars <- setdiff(colnames(df), drop_vars)
df <- df[, .SD, .SDcols=keep_vars]
df[, deceased_or_mrsgeq3:=deceased | mrsendofstudy>=3]
df[, deceased_or_mrsgeq4:=deceased | mrsendofstudy>=4]

# drop 2 observations with missing country
df <- df[!is.na(country)]
df[, table(country, earlyrest)]

# summarize countries by fraction of early restriction
dfc <- df[, list(frac_earlyrest=mean(earlyrest)), by="country"]
df[dfc, restcountry:=i.frac_earlyrest >= .05, on="country"]

fwrite(df, here::here("data", "df_curated.csv"), row.names=F)

