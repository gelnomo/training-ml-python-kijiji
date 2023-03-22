# <font color="#00ced1"><b>Kijiji Car Sales Ads</b></font>
The global connection to the internet has become an essential part of our daily lives, allowing greater ease and accessibility in purchasing, selling, and exchanging services. There are many virtual platforms that have been built and positioned as references in the subject. One of these companies is Kijiji, founded in 2005 by Janet Bannister in Canada as a local classifieds platform (Kumaran, M., 2016). Over the years, it has grown to become one of the largest online marketplaces in Canada, offering a variety of goods and services to its users. In 2007, it was acquired by eBay, allowing it to expand its reach and continue to grow. (eBay Inc. Staff, 2014) Today, Kijiji has millions of users and remains a trusted and convenient platform for buying and selling goods and services.

## Table of contents <a class="anchor" id="0.1"></a>

1. [Business Case Evaluation](#1)
2. [Data Identification](#2)
   1. [Metadata](#2.1)
3. [Data Acquisition & Filtering](#3)
4. [Data Analysis](#4)
10. [Conclusion](#5)

>**Group 6**
>- Camilo Florez (888810)
>- Yasar Mansuri (884738)
>- Syed Zumair (885081)
>- Erwin Mongui (883945)


[Back to Table of Contents](#0.1)
## Business Case Evaluation <a class="anchor" id="1"></a>

### Problem definition
Kijiji company requires predict car visits depending on the data recollected.

[Back to Table of Contents](#0.1)
## Data Identification <a class="anchor" id="2"></a>
The data contains information related to Kijiji Car sales ads. Available data for more than 51,000 ads on the Kijiji website was collected by web scraping on August 27, 2022 <br><br>

|COLUMN| DESCRIPTION                                                                                                                                                                                                |
|---|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|AdID| Unique ad value. It is helpful to identify the specific item                                                                                                                                               |
|city| List of cities where the ad is going to appear                                                                                                                                                             |
|latitude| Y point distance where the ad is located                                                                                                                                                                   |
|longitude| X point distance where the ad is located                                                                                                                                                                   |
|mapAddress| Ad address/postal code where the ad is located                                                                                                                                                             |
|title| It is the Ad heading compound by year and model vehicle                                                                                                                                                    |
|price| Sell car price                                                                                                                                                                                             |
|saleby| Who owns the ad                                                                                                                                                                                            |
|make| Brand of the vehicle                                                                                                                                                                                       |
|model| Specific version on the vehicle                                                                                                                                                                            |
|Trim| Pre-packaged groups of features for a particular car                                                                                                                                                       |
|year| The actual year when the car was built                                                                                                                                                                     |
|condition| used, damaged, new, salvage, lease_takeover                                                                                                                                                                |
|Kilometers| It is the mileage that determines what a used vehicle will sell for. A general starting point to consider is 20,000 KM or less per year                                                                    |
|body| The super-structure for all vehicles                                                                                                                                                                       |
|Drivetrain| It is a mechanical system that transfers energy from the engine to the wheels to make them move. four-wheel drive (4WD)/4x4, Rear-Wheel Drive (RWD), FWD (Front-Wheel Drive), all-wheel drive (AWD), other |
|fueltype| gas, diesel, other, hybrid, and electric                                                                                                                                                                   |
|color| Car color                                                                                                                                                                                                  |
|Date| ad publishing date                                                                                                                                                                                         |
|visit| total number of visits                                                                                                                                                                                     |
|No of door| Number of doors                                                                                                                                                                                            |
|No of seats| Number of seats                                                                                                                                                                                            |
|Transmission| 0: None, 1: Manual Transmission (MT), 2: Automatic Transmission (AT), 3: Other                                                                                                                             |
|push button start| 1: Included, 0: Not included                                                                                                                                                                               |
|parking assist| 1: Included, 0: Not included                                                                                                                                                                               |
|sunroof| 1: Included, 0: Not included                                                                                                                                                                               |
|alloy wheels| 1: Included, 0: Not included                                                                                                                                                                               |
|VIN| It is a 17 digits number that uniquely identifies the car. It can be found on the driver’s side dashboard of your vehicle.                                                                                 |

[Back to Table of Contents](#0.1)
## Metadata <a class="anchor" id="2.1"></a>

- Source: Dataset came from Kaggle: [Dataset: kijiji-car-sales-ads](https://www.kaggle.com/datasets/hossaingh/kijiji-car-sales-ads?select=AdData.csv)
- Collaborators: Hossain (Owner)
- Coverage:
    - Temporal Coverage end data: 08/27/2022
    - Geospatial Coverage: Canada
- Provenance: Available data for more than 51,000 ads on the Kijiji website was collected by web scraping on August 27, 2022.
- License: CC BY-NC-SA 4.0

[Back to Table of Contents](#0.1)
## Data Acquisition & Filtering <a class="anchor" id="3"></a>
Review data
> According to the definition of each column, there is a general approach to the content of the data. It is possible to predict the price depending on the car's features.

Identify valuable information
> Relevant information is price, city, make, model, Trim, year, condition, Kilometers, body, Drivetrain, fuel type, No of the door, No of seats, and Transmission. According to the problem definition, the following process will predict the price depending on different filters.


[Back to Table of Contents](#0.1)
## Data Analysis <a class="anchor" id="4"></a>

Create environment

```bash
python3 -m venv env
. env/bin/activate
```

Install libraries

```bash
pip install -r requirements.txt
```

Run the project

analyze: console/file. Generate the summary in a file or console
training: console/file. Generate the summary in a file or console, it's required analyze to use this command
normalization: yes/no. Generate all analysis and training with a normalization data
storage_model: yes/no. Save each model in a joblib file to test on unittest section

```bash
python3 main.py --analyze console --training file --normalization no
```

For the realization of the extraction and transformation of data began by selecting the libraries that would be used in the process, within these are: Numpy, Matplotlib, Pandas, Skelearn, Scipy and seanborn.  To understand the data, the shape of the dataset is 51343, 28 and the Date column is imported as a datetime data type.
As a first initial step, an analysis is performed to identify min, max, standard deviation, and quartiles values for each ordinal and categorical column. Some columns, such as price, year, and kilometers, are equivalent to the initial shape. The trim, title, drivetrain, fuel type, color, number of doors, and number of seats columns contain fewer values than the initial form. The average price is 37,198 for each car make. However, the minimum and maximum prices show some outliers that could affect the average.  There is also evidence of values of the type of Nan and Null that must be cleaned. 

The following graph shows the percentage of null in the dataset. It is visible that some columns, such as Latitude, Longitude, mapaddress, and VIP, have no data at all.

![heatmap_nulls_quantity_nulls_per_column.png](reports%2Ffigures%2Fheatmap_nulls_quantity_nulls_per_column.png)
![bar_plot_percentage_nulls_per_columns.png](reports%2Ffigures%2Fbar_plot_percentage_nulls_per_columns.png)

The initial correlation is created to understand which fields are valuable for solving the problem definition (price). Initially, it is notable that year, parking assist, alloy wheels, and transmission have a bit of relevance. Nevertheless, the other columns are categorical. Those could be more accurate once they become to numerical. As a result, it is necessary cleaning the data.

![heat_map_corr_correlation_before_cleaning.png](reports%2Ffigures%2Fheat_map_corr_correlation_before_cleaning.png)


### Graph: Number of Ads by City
This graph shows that the majority of the ads were published in Calgary, with 16,612 ads, in contrast with other cities.

![count_bar_plot_number_of_ads_by_city.png](reports%2Ffigures%2Fcount_bar_plot_number_of_ads_by_city.png)

### Graph: Number of Ads by Year
This graph shows that the majority of the ads were published in 2022, with 47,546 ads, in contrast with the previous year, with 1,733 ads

![count_bar_plot_number_of_ads_by_year.png](reports%2Ffigures%2Fcount_bar_plot_number_of_ads_by_year.png)

### Graph: Number of Ads by Month
This graph shows that the majority of the ads were published in August, with 20,847 ads, in contrast with the other months. As a hypothesis, it could be related to the Summer season.

![count_bar_plot_number_of_ads_by_month.png](reports%2Ffigures%2Fcount_bar_plot_number_of_ads_by_month.png)

### Graph: Number of Visits by Month
This graph shows that July and August are the great months in 3 years with more than 3,000,000 visits each month, in contrast with the other months.

![bar_plot_number_of_visits_by_month.png](reports%2Ffigures%2Fbar_plot_number_of_visits_by_month.png)

### Graph: Number of Visits by Month and Year
This graph shows that 2022 increased the visits compared to the previous years in City of Toronto and Calgary. 

![line_plot_monthly_visits_by_year.png](reports%2Ffigures%2Fline_plot_monthly_visits_by_year.png)

### Graph: Number of Visits by Year by City
This graph shows that Ford has been one of the more searched cars in the last two years.
![bar_plot_visits_by_year_by_city.png](reports%2Ffigures%2Fbar_plot_visits_by_year_by_city.png)

### Graph: Media Visit by Year (Make Car)
This graph shows that Ford, Chevrolet, and Toyota are more searched on Kijiji in the last two years.
![bar_plot_make_(visit)_by_year.png](reports%2Ffigures%2Fbar_plot_make_%28visit%29_by_year.png)

### Graph: Shared-of-Visit by Year
This graph shows the distribution in the last years, where 2022 year got 82% of the visits

![count_pie_plot_share-of-visit_by_year.png](reports%2Ffigures%2Fcount_pie_plot_share-of-visit_by_year.png)

### Identify Outliers
In the next step, it is going to identify prices that seem to be errors with the Z score method.  The Z score method is used to determine how many standard deviations a data point is from the mean. Roughly 68%, 95%, and 99.7% of the data points fall within 1, 2, and 3 standard deviations from the mean, respectively. Data points with a Z score greater than 3 are considered outliers.
![distribution_kdeplot_before_-_visits_density.png](reports%2Ffigures%2Fdistribution_kdeplot_before_-_visits_density.png)

As the last step of data preprocessing, all the branded cars were collected in different data sets stored in a dictionary to be processed one by one. Categorical fields were converted into numerical (fuel type, condition, model, city, transmission) and unnecessary fields such as visits, kilometers, cyear cmonth cmonthname.	
![distribution_kdeplot_after_-_visits_density.png](reports%2Ffigures%2Fdistribution_kdeplot_after_-_visits_density.png)

As a result of cleaning the outliers the price density changed. It is notable that a lot of prices are distributed between $100 to $100,000.

### Covariance
- Slope is negative between visits vs. price.
- A negative covariance reveals that there is an inverse relationship between the variables, that is, as one increases, the other tends to decrease.
- If the ad price goes up, the number of visits will reduce.

![count_scatter_correlation_month.png](reports%2Ffigures%2Fcount_scatter_correlation_month.png)
![count_lmplot_correlation_price_per_year.png](reports%2Ffigures%2Fcount_lmplot_correlation_price_per_year.png)

### Train/Test Data
Once the preprocessing steps were completed, the dataset was divided into two sets: a training set (60% of the data) and the test dataset was taken directly from the data frame, for each publication, all the models were applied and the model with the best performance was chosen.
-	Train Data = 0.6 (60%) 30000 rows
-	Test Data = 0.4 (40%) 20000 rows
-	Radom State = 42

### Regressor choice of ML algorithm
The research is done to identify different models that solve the problem definition. As a result, some models that are focused on predicting prices are Liner Regression, Gradient Boosting Regressor, and Random Forest algorithms. To get a better price for each combination, the aim is to get used the models between each other to have the most accurate value.

### Why algorithm was chosen
It was decided to use three regression algorithms to build the price prediction model. The reasons for the use of these algorithms are that they are techniques used for the prediction of continuous variables as is the current case of the price. The model was built using the Scikit-learn library in Python and the goal is to compare the accuracy of each model by making a comparison of the error of each in the prediction. From the research on the internet, as analyzed some real-life use cases, and from that, most of the price is predicted by using the linear regression and random forest algorithm for higher accuracy and less error. For boosting accuracy, the gradient boosting algorithm is used. (Zakariya, M., 2022)

### How the models work

#### Linear Regression
It is a statistical technique that seeks to find the linear relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the input and output variables and seeks to fit a line that best describes the relationship. Linear regression is often used when the relationship between the variables is linear, and the goal is to make predictions within the range of the data.

#### Random Forest Regressor
It is a supervised learning algorithm that builds a set of decision trees and aggregates their results to produce a final output. It creates a forest of decision trees by randomly selecting samples from the data and features to build each tree. Random forest is often used when the relationship between the variables is non-linear, and the goal is to make predictions over a broader range of data, including out-of-sample data.

#### Gradient Boosting
It is a machine learning algorithm that combines several weak models to create a strong model. The algorithm works by iteratively adding new models to the ensemble, each one correcting the errors made by the previous models.

### Evaluating the Best Result (MSE, MAE, R squared, etc.)
Predicting the accurate value requires choosing the ideal model for each combination collected or compared. One way to identify the less risk is by validating the error generated by each model. Root mean square error (RMSE) is checked to recognize how far the data points are from the regression line. With a lower number in RMSE has an excellent performance.
There are a lot of errors between each model; identify the percentage of the model used to predict the dataset. It was RFR (Random Forest Regressor) with a 50%.

### Analyzing the Best Result
After comparing the real price vs. the price predicted by each model, it is shown that the price expected has similar behavior to the real ones, which means that the project had a good result; however, in the car models from 1940 to 1960, the data require more cleansing in order to improve the prediction.

![comparison_plot_rfr_mean_visit_vs_mean_visit_predicted.png](reports%2Ffigures%2Fcomparison_plot_rfr_mean_visit_vs_mean_visit_predicted.png)
![comparison_plot_lr_mean_visit_vs_mean_visit_predicted.png](reports%2Ffigures%2Fcomparison_plot_lr_mean_visit_vs_mean_visit_predicted.png)
![comparison_plot_gbr_mean_visit_vs_mean_visit_predicted.png](reports%2Ffigures%2Fcomparison_plot_gbr_mean_visit_vs_mean_visit_predicted.png)
![comparison_plot_mean_visit_vs_mean_visit_predicted.png](reports%2Ffigures%2Fcomparison_plot_mean_visit_vs_mean_visit_predicted.png)

[Back to Table of Contents](#0.1)
## Conclusion <a class="anchor" id="5"></a>
During the analysis, it was notable that prices and visits change depending on factors such as season and internal company strategies. Despite this, it was processed for a model to predict prices. It was based on different training to get a precise value. The project was not found on one algorithm but also on others, such as Linear Regression, Random Forest Regression, and Gradient Boosting Regressor, to get the best price. Some results show that Random Forest Regression is more used to get an accurate price.
The model takes 30 min to be trained. As a result, the result is stored in a data frame. This info can be stored on an SQL database or NoSQL to be accessible to an API. It is suggested that the training can be each five months because the prices will change. The goal is to have a model the most updated as possible.
