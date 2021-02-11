"""Goal: Decompose CO2 Emissions by fuel and sector (historical and projected) 

Required inputs: CO2 Emissions data (EIA)
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


Desired outputs: results of decomposition in csv and visualizations (as in the rest of EII)
        
"""