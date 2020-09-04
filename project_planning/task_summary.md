# Task/Issue Breakdown
Purpose is to identify the necessary components of re-creating the PNNL spreadsheets' approaches to collecting activity and energy data, manipulating collected activity and energy data, and calculating the final LMDI results by end use sector. This list will be translated into issues and integrated with [GitHub Projects](https://github.com/NREL/EnergyIntensityIndicators/projects/1).

## 1. Documentation
Sphinx documentation based on docstrings.

What are the remaining tasks/issues?

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
### 3a. EIA API
* [x] SEDS. Implemented with `pull_eia_api`.
    - SEDS by Census Region from use_all_btu.csv
* [x] AER. Implemented with `pull_eia_api`.
* [x] SEDS. Implemented with `pull_eia_api`.
* [x] Conversion Factors. Implemented with `pull_eia_api`
* [X] National Calibration. Implemented with `pull_eia_api`
* [] Weather factors. Implemented with `weather_factors`
#### 3.a.i. 
* [] Calculate LMDI for Prices (used in weather factors)    

### 3b.Census Bureau API
* [ ] Annual Survey of Manufacturers
* [ ] Economic Census

### 3c. Transportation Energy Databook (downloaded excel file)


## 4. Data Processing
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
* [] Activity data: Floorspace estimates
    - Historical Floorspace --> CO-StatePop2.xls
    - Regional Floorspace
    - Regional Shares
        - CBECS Data
        - Residential Sector Final Floorspace Estimates

### 4c. Transportation
* [] Passenger based activity
* [] Passenger based energy use
* [] Freight based activity
* [] Freight based energy use

### 4d. Industrial

### 4e. Electricity
* [] Reconcile physical units
* [] Create Energy and Activity dataframes

### 4f. Economywide
