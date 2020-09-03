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
* [x] AER. Implemented with `pull_eia_api`.
* [x] SEDS. Implemented with `pull_eia_api`.

### 3b.Census Bureau API
* [ ] Annual Survey of Manufacturers
* [ ] Economic Census

### 3c. Transportation Energy Databook (downloaded excel file)


## 4. Data Processing
Processing data from collection to input for LMDI calculations
### 4a. Residential

### 4b. Commercial

### 4c. Transportation

### 4d. Industrial

### 4e. Electricity

### 4f. Economywide
