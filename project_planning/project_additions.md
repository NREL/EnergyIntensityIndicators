
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
## Goal
- 

## Required inputs:
- 

## Framework:
- Use minimum and maximum NAICS digits for level of aggregation (e.g. minimum of 3 and maximum as high as each MECS year allows)

## Steps:
- Explore highest number of NAICS digits contained in each MECS year (differs by year)

## Desired outputs:
- Standard EII outputs for the manufacturing sector (with flexible levels of aggregation)



# Trade Flow Decomposition

## Goal: 
- Incorporate embodied energy from imported goods in decomposition of the Industrial Sector. 

## Required inputs:
- BEA industry accounts data (trade extensions broken into intermediate/final goods)

## Framework: 
- Net out intermediate inputs imported


## Steps:


## Desired outputs: 
- results of decomposition in csv and visualizations (as in the rest of EII) where imports are a sub-sector level?

## Resources:
- [Offshoring Bias](https://www.federalreserve.gov/pubs/ifdp/2010/1007/ifdp1007.htm)
- [Measuring Globalization](https://research.upjohn.org/up_press/232/)
- [Intermediate Inputs and Economic Productivity](https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.2011.0565)
- [Industrial Energy Data Book](https://www.nrel.gov/docs/fy20osti/73901.pdf)