# Calculation Approach
The Energy Intensity Indicators are a series of [log mean divisia index (LMDI)](https://doi.org/10.1016/j.enpol.2003.10.010) decompositions implemented hierarchically by economic sector (residential, commercial, transportation, industry, and electric).

Each LDMI hierarchy level uses shared calculations, performed on activity and energy data of that level. Portions of the results are then inherited by the next level of aggregation.

## Shared Calculations (work in progress)
These are the operations that are common across (most) economic sectors. Dependencies for data or prior calculations are indicated in parenthesis.

### Energy Data
*Commercial and residential data include weather adjustment*
*Calculations performed for site, source, elect, fuels for all sectors but transportation and electricity*
* Energy Shares(Energy Data)
    * Log-Mean Weights(Energy Shares)
        * Log-Mean Divisia Weights(Log-Mean Weights)

### Activity Data
* Activity Shares(Activity Data)
    * Log Changes-Activity Shares(Activity Shares)


### Other Indices
* Nominal Intensity Index(Energy Data, Activity Data):
*Commercial and residential data include weather adjustment*
    * Energy Intensity Index(Base Year, Nominal Intensity Index)
        * Log Changes-Intensity(Energy Intensity)
            * Computed Intensity Index(Log-Mean Divisia Weights, Log Changes-Intensity)
            * Computed Structure Index(Log-Mean Divisia Weights, Log Changes-Activity Shares)

### Final LDMI Components
* Activity index(Base Year, Activity Data)
* Index of Aggregate Intensity(Base Year, Energy Intensity Index)
* Structure Index(Computed Structure Index)

## Inherited Results
There are the results of a lower-level LMDI decomposition that are used in the calculation of a higher-level LMDI decomposition.
* Energy Intensity Index?
* Computed Structure Index
