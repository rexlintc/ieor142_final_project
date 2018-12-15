# import necessary libraries
library(ggplot2)
library(pitchRx)

# scrape data
baseball_data <- scrape(start="2007-11-01", end="2010-11-01")

# convert data into data frames
atbat <- as.data.frame(baseball_data$atbat)
action <- as.data.frame(baseball_data$action)
pitch <- as.data.frame(baseball_data$pitch)
po <- as.data.frame(baseball_data$po)
runner <- as.data.frame(baseball_data$runner)

# clean dataframe and get features
data <- data.frame("id"=pitch$id, "pitch_id"=pitch$sv_id, "inning"=pitch$inning, 
                "x55"= pitch$x0, "y55"= pitch$y0, "z55"= pitch$z0, "vx55"= pitch$vx0 , 
                "vy55"= pitch$vy0, "vz55"= pitch$vz0, "ax"= pitch$ax , "ay"= pitch$ay, 
                "az"= pitch$az, "start_speed"= pitch$start_speed, 
                "end_speed"= pitch$end_speed, "spin_rate"= pitch$spin_rate, 
                "spin_axis"=pitch$spin_dir, "pfx_x" = pitch$pfx_x, "pfx_z"= pitch$pfx_z,
                "type" = pitch$pitch_type, "break_x"= pitch$break_y, 
                "break_x"=pitch$break_angle, "num"=pitch$num, "zone" = pitch$zone, 
                "release_x" = pitch$sz_top, "release_y" = pitch$sz_bot, "inning"=pitch$inning)
pitcher_data = data.frame("pitcher"= atbat$pitcher, "outs" = atbat$o, "balls" = atbat$b,
                          "strikes" = atbat$s, "batter_side" = atbat$stand, 
                          "batter" = atbat$batter, "num" = atbat$num, 
                          "pitcher_side" = atbat$p_throws, "next" = atbat$next_, 
                          "batter_height" = atbat$b_height)
data$pitch_speed = (data$start_speed + data$end_speed)/2

# merge pitch data and pitcher and batter information
data <- merge(data, pitcher_data, by="num")

# export to csv
write.csv(data, file="PitchData.csv")

