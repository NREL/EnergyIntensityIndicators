# [OpenEI](https://openei.org/wiki/Main_Page) Data Visualization Generator – Data Ingestion Methodology

After some discussion, it became evident that the column naming convention would be the key component to the development of a flexible, ingestible data model.  We believe the following column naming convention will afford the tool the most flexibility while meeting the needs for data classification including identification of the data type, categories, label, and units to the data value, which will be stored in rows under the column header.

Column header key:
`@type|Category1|Catergory2|...|Label#units`

The data ingestion script reads each column header and creates the necessary type classification, categories, labels, units, etc. Each data type may have required fields such as “Label” or “units.” The script detects any errors in the column heading structure and warns the user to help ensure compliance with each format. This format provides flexibility, and users will be able to add multiple levels of categories and sub-categories to meet their specific data needs.

The following data types have been proposed (an ellipsis ... indicates an optional parameter):

`@filter|Category1|...Category2|...|Label#units`
-	A list of options that can be grouped by 1 or more categories.

`@weight|Category1|...Category2|...|Label#units`
-	A weighted value to use with a matching filter (must match filter label and categories).

`@scenario|Label`
-	A list of options that are completely separate from each other, i.e. they will not be seen on the same chart at the same time. The options come from the unique values in the scenario column.

`@timeseries|Label`
-	A list of options that can be used to make a time series, e.g. a list of years.

`@geography|Label`
-	A list of geography names, e.g. states, counties, cities, that can be used in charts or a choropleth map.

`@geoid`
-	The column values are geography IDs that can be used in a choropleth map.

`@latlong`
-	Latitude and longitude coordinates
## Example
A user could create the following column headers:  
`@filter|Generation|Renewable|Solar#Twh`  
`@filter|Generation|Renewable|Wind#Twh`  
`@filter|Generation|Non-Renewable|Coal#Twh`  
`@filter|Generation|Non-Renewable|Natural Gas#Twh`

This would structure the inputs in the following way:
Generation(Twh)
*	Renewable
    -	Solar
    -	Wind
*	Non-renewable
    -	Coal
    -	Natural Gas

## Input Format
The data ingestion will utilize comma separated value (CSV) spreadsheets rather than Excel files. While powerful, Excel files are more complicated to parse and do not add any significant advantages to this data ingestion model. CSV files are an easy-to-use format than can be opened and modified in many different programs without risking formatting or data corruption issues that may arise from Excel files.

## Conclusion
We feel this column naming convention will support the development of a flexible tool while being simple enough for novice users and our initial prototype.
