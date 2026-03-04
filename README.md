# Aqi-Reflex-Agent - Project 'ARA'

This is the link to our website. We have given the Python Program a User Interface to better communicate with the user.

https://aqi-reflex-agent.netlify.app/

**Due to Free Tier Services our website may take about a minute to load if the server has put the Python Backend Framework to sleep.**
**To Force Start the Server Visit this link**
https://aqi-reflex-agent.onrender.com/

--------------------------------------------------------------------------------------------------------------------------------------------

#  Tech Stack

1. Python3.
2. HTML CSS JS
3. Netify for front-end Hosting
4. Flask Library.
5. Scikit learn (Random Forest Regressor) Library.
6. Gunicorn (For Render To host Backend).
7. Pandas Library for CSV data Handling.

--------------------------------------------------------------------------------------------------------------------------------------------

# The CSV Dataset

Our agenet uses a **merged annual dataset of environmental and air quality indicators** across multiple U.S. locations. The dataset combines pollutant measurements, geographic information, and climate-related variables to support environmental analysis, pollution monitoring, and data science research.

## Dataset Overview

* **File:** `merged_annual_data.csv`
* **Rows:** 15,192 records
* **Columns:** 14 features
* **Granularity:** Annual data per city/location

Each row represents **one city's environmental metrics for a specific year**.

## Features (Columns)

| Column          | Description                                  |
| --------------- | -------------------------------------------- |
| **State Name**  | U.S. state where the data was recorded       |
| **County Name** | County within the state                      |
| **City Name**   | City where measurements were taken           |
| **Latitude**    | Geographic latitude of the location          |
| **Longitude**   | Geographic longitude of the location         |
| **Year**        | Year of the recorded environmental data      |
| **CO**          | Carbon Monoxide concentration                |
| **NO2**         | Nitrogen Dioxide concentration               |
| **Temperature** | Average temperature for the year             |
| **Ozone**       | Ozone concentration levels                   |
| **PM2.5**       | Fine particulate matter (particles ≤ 2.5 µm) |
| **Humidity**    | Average humidity levels                      |
| **SO2**         | Sulfur Dioxide concentration                 |
| **Wind Speed**  | Average wind speed                           |

## Potential Use Cases

* Air pollution trend analysis
* Environmental impact studies
* Climate and weather correlation analysis
* Geographic pollution mapping
* Machine learning models for air quality prediction

## Notice

* Some values may contain **missing data (NaN)** depending on sensor availability or reporting gaps.
* Data is aggregated **annually**, not daily or hourly.

## Example Record

| State   | County  | City     | Year | Ozone  | PM2.5 |
| ------- | ------- | -------- | ---- | ------ | ----- |
| Alabama | Baldwin | Fairhope | 2015 | 0.0408 | 8.63  |


--------------------------------------------------------------------------------------------------------------------------------------------

# Precision and Error Analysis

## Precision

The AQI our Agent Predicted for Wisconsin - Milwaukee

<img width="466" height="744" alt="image" src="https://github.com/user-attachments/assets/dd8ce144-f1c8-4094-be6d-4e9778e8a4f6" />

The AQI from **https://www.aqi.in/in/dashboard/united-states/wisconsin/milwaukee** at 3rd March 2026 21:31 Hours IST

<img width="1159" height="393" alt="image" src="https://github.com/user-attachments/assets/b5fda596-4606-4224-a054-857f482355af" />

## Error Analysis

Percentage Error using the formula

(AQI<sub>agent</sub> - AQI<sub>Website</sub>)/AQI<sub>Website</sub>

Gives us about 45.88% Error for such low readings.

This still gives us accurate air pollution level readings as per US-EPA 2016 standards

| AQI Range | Air Pollution Level                |
|-----------|------------------------------------|
| 0–50      | Good                               |
| 51–100    | Moderate                           |
| 101–150   | Unhealthy for Sensitive Groups     |
| 151–200   | Unhealthy                          |
| 201–300   | Very Unhealthy                     |
| 300+      | Hazardous (Delhi Air ;0)           |

--------------------------------------------------------------------------------------------------------------------------------------------
## License and Bibliography

This dataset is provided for **research and educational purposes**.
This Dataset has been Picked from **"Kaggle"**














