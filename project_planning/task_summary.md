# Task/Issue Breakdown
Purpose is to identify the necessary components of re-creating the PNNL spreadsheets' approaches to collecting activity and energy data, manipulating collected activity and energy data, and calculating the final LMDI results by end use sector. This list will be translated into issues and integrated with [GitHub Projects](https://github.com/NREL/EnergyIntensityIndicators/projects/1).

## 0. Overall Organization
1. Historical code files
    * Residential.py
    * Commercial.py
    * Industrial.py
    * Electricity.py
    * Economy-wide.py
2. Projection code files
    * Residential_projection.py
    * Commercial_projection.py
    * Industrial_projection.py
    * Electricity_projection.py
3. `LMDI.py` (includes both multiplicative and additive forms, for now.)
4. API code files
    * pull_eia_api
    * pull_...
5. Summarization code files
    * NREL data viz
    * Summary charts and tables

## 1. Documentation
Sphinx documentation based on docstrings.

What are the remaining tasks/issues?
* [] Reference each script (high level outline)
* [] Edit/clean up docstrings

## 2. LMDI
Shared classes/methods for end-use sector decomposition
### 2a. Multiplicative Form (PNNL method)
* [x] Calculates the log changes to intensity. Implemented with `outline.calculate_log_changes`.
* [x] Adjust source energy. Implemented with `outline.get_source_adj`.
* [ ] ...

### 2b. Additive Form (new)
* [ ] Function 1
* [ ] Function 2

## 3. Data Collection
Processes for automated (where possible) retrieval of primary activity and energy data.
**Historical data can be imported as a csv as long as there aren't any data manipulations (e.g., interpolations, curve fitting) that aren't updated with new data.**
### 3a. EIA API
* [x] Get EIA Data with `pull_eia_api` eia_api
* [x] SEDS. Implemented with `pull_eia_api`.
    - SEDS by Census Region from use_all_btu.csv
* [x] AER. Implemented with `pull_eia_api` \ with read_csv and url
* [x] Conversion Factors. Implemented with `pull_eia_api`
* [X] National Calibration. Implemented with `pull_eia_api`.- fill values
* [x] Weather factors. Implemented with `weather_factors`
#### 3.a.i.
* [] Calculate LMDI for Prices (used in weather factors)    

### 3b. Residential
* [ ] American Annual Housing Survey
    * [] Housing Stock Model
* RECS
    * [ ] Historical:
    * [ ] Updates:

### 3c. Commercial
* CBECS
    * [x] Historical: hard coded
    * [ ] Verify if calculations are updated with new CBECS

### 3d. Industry
* Census Bureau API
    * [ ] Annual Survey of Manufacturers (CM to send starting points)
    * [ ] Economic Census (CM to send starting points)

* Bureau of Economic Analysis
    * [x] Chained and quantity indexes for value added and gross output. Implemented with `pull_bea_api.py`
* Miranowski data for agriculture
* MECS
    * [ ] Historical: hard code as csv from PNNL
    * [ ] Updates: create method for manual download


### 3e. Transportation
* [ ] Transportation Energy Databook: downloaded excel file; use for updating indicators.
* [ ] Create summary historical csv. Document sources for each piece of data in dosctring of method for importing csv. Sources include:  
    * [] American Public Transit Association
    * [] FWHA


## 4. Data Processing (Historical [PNNL])
Processing data from collection to input for LMDI calculations

### 4a. Residential
* [x] Energy data:
    - from SEDS implemented with `pull_eia_api`
* [] Activity data:
    - Occupied Housing Units
    - Floorspace (Final Floorspace Estimates)
        - Total Stock SF: implemented in `GetCensusData.get_housing_stock` and `GetCensusData.final_floorspace_estimates`
        - Total Stock MF: implemented in `GetCensusData.get_housing_stock` and `GetCensusData.final_floorspace_estimates`
        - Total Stock MH: implemented in `GetCensusData.get_housing_stock` and `GetCensusData.final_floorspace_estimates`
        - Calculated Shares by Region (from AHS tables)
        - Ratios to National Average Size (from AHS tables)


### 4b. Commercial
* [] Energy data: adjusted supplier data
    - Uses data from AER11 Table 2.1, National Calibration, EIA via Survey EIA-861
    * [] Get EIA Survey 861 Data (Sectoral_reclassification5.xls  (10/25/2012))
        * [] create method for adjustment with if statement for year (e.g. if year > 2009, do this calculation)
* [] Activity data: Floorspace estimates (don't want this array hard coded)
    - Historical Floorspace --> CO-StatePop2.xls
    - Regional Floorspace
    - Regional Shares
        - CBECS Data
        - Residential Sector Final Floorspace Estimates
    - NEMS logistic

### 4c. Transportation
* [] Create Passenger based activity dataframe
* [] Create Passenger based energy use dataframe
* [] Create Freight based activity dataframe
* [] Create Freight based energy use dataframe
     * [] ENO

### 4d. Industrial
* [] Manufacturing Data
* [] Agriculture Data
* [] Mining Data
* [] Construction Data

### 4e. Electricity
* [] Reconcile physical units
* [] Create Energy and Activity dataframes

### 4f. Economy-wide

## 5. Data Processing (Projection)
All data come from EIA API for AEO (category_id=964164).
**Will need to identify AEO data that are analogous to data used for historical projections.** Started in `projections.py`, which maps API series_ids.
### 5a. Residential
AEO projects annual estimates for cooling degree days.


* [] Energy data:
* [] Activity data:
    - Occupied Housing Units
    - Floorspace (Final Floorspace Estimates)

### 5b. Commercial
* [] Energy data: adjusted supplier data
* [] Activity data


### 5c. Transportation
* [] Energy data: adjusted supplier data
* [] Activity data

### 5d. Industrial
* [] Energy data: adjusted supplier data
* [] Activity data
* [] Manufacturing Data
* [] Agriculture Data
* [] Mining Data
* [] Construction Data

### 5e. Electricity
* [] Reconcile physical units
* [] Create Energy and Activity dataframes

### 5f. Economy-wide
