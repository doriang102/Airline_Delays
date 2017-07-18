# Predicting Airline Delays




For this analysis, I will consider the 2016 airline delay data only, and my goal will be to predict delays using this one year of data. 

I have several hypotheses which I will evaluate throughout this presentation.

**Goal:** *Given all of the airline data from 2016, can we predict delays?*
I tried this using two methods:

1. Regression on the departure delay.
2. Classification above a 20 minute threshold (defined as a delay by most major airlines).

I used the following data in my analysis.

**Data:**

- Custom built Poisson regression model for capturing hourly and daily trends. 
- The 2016 weather data obtained from [NOAA](https://www.ncdc.noaa.gov/).
- Data regarding the age and model of the plane obtained from [stat-computing](http://stat-computing.org/dataexpo/2009/plane-data.csv).
- The total capacity of the airport from which the airlplane is originating form [Wiki](https://en.m.wikipedia.org/wikiList_of_the_busiest_airports_in_the_United_States.)
- USA Holidays.

**Main Results:**

![alt text](fig/roc_final.png)






 