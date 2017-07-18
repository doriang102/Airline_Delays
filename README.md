# Predicting Airline Delays



### Introduction

For this analysis, I will consider the 2016 airline delay data only, and my goal will be to predict delays using this one year of data. 

I have several hypotheses which I will evaluate throughout this presentation.

**Goal:** *Given all of the airline data from 2016, can we predict delays?*
I tried this using two methods:

1. Regression on the departure delay.
2. Classification above a 20 minute threshold (defined as a delay by most major airlines).

I used the following data in my analysis.

### Data sets:

- [Airline On-Time Performance Data. RITA/BTS. Bureau of Transportation Statistics.]( https://www.transtats.bts.gov). 2016.

- [Local Climatological Data. National Centers for Environmental Information.]( https://www.ncdc.noaa.gov/cdo-web/datatools/lcd). 2016.

- [Flight Standards Service — Civil Aviation Registry. Federal Aviation Administration.]( http://stat-computing.org/dataexpo/2009/plane-data.csv). 2009.

- [Passenger Boardings at Commercial Service Airports. Federal Aviation Administration.] (https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/media/cy14-commercial-service-enplanements.pdf). 2014.  

*Caveats of above data:* Plane data is from 2009, but will have missing planes from 2010-2016. 

### Main Results:

#### Performance:

We compare performance of multiple models with and without the additional data used above. More precisely, we define:

**Original:** Only the Airline On-Time Performance Data.

**Enhanced:** All of the data included in the above links, in addition to a customized Poisson variable.

$\text{S}_1(N) = \sum_{p=1}^N \text{E}(p)$

Below we see a comparison of three different classification models for the original and enriched variable set. The best performance obtained was by the Random Forest Classifier with the enriched data set, which achieved an ROC of 0.76.


![alt text](fig/roc_final.png)

#### Variable Importances:

**Gradient Boosted Trees:**

![alt text](fig/gb_variables.png)


**Random Forest:**

![alt text](fig/rf_variables.png)


 