
# GHG Decomposition
## Goal: 
- Decompose CO2 Emissions by fuel and sector (historical and projected) 

## Required inputs: 
- CO2 Emissions data (EIA):
    Notes:
        -EIA API contains endpoints for CO2 emissions by state 
        (including total US), fuel and sector (including all)
            ~ Fuels (depend on sector?): 
                - All fuels
                - Coal
                - Natural Gas
                - Petroleum
        - Link: https://www.eia.gov/opendata/qb.php?category=2251604
        - AEO (projections) includes Energy-Related Carbon Dioxide Emissions by 
        Sector (including total), Source, and Region (including total US)
            ~ Sources (depend on sector): 
                - Coal
                - Natural Gas
                - Petroleum
                - Electricity
                - Total
- CO2 Emissions factors (EPA):
        - https://www.epa.gov/climateleadership/ghg-emission-factors-hub
    
## Framework: 
- Multiply fuel use by emissions factor? (using constant factors over time)
- Replacing energy use with emissions in decomp formula
- plot difference between EIA emissions and calculated emissions

## Steps: 
- Import collect_data methods from each sector specific class
- collect emissions factor data (from EIA?)
- multiply energy use data by emissions factor data, replace energy use data 
    with the product (emissions by energy type)
- pass resulting data dictionaries to CalculateLMDI class

### Notes:
- This flow follows similar process to EconomyWide, potentially inherit that class

## Desired outputs: 
- results of decomposition in csv and visualizations (as in the rest of EII)
        





# Manufacturing Sector Expansion




# Trade Flow Decomposition

## Goal: 
- Incorporate embodied energy from imported goods in decomposition of the Industrial Sector. 

## Required inputs:


## Framework: 

## Steps:


## Desired outputs: 
- results of decomposition in csv and visualizations (as in the rest of EII) where imports are a sub-sector level?