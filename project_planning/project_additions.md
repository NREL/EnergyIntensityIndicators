
Isabelle's last day is 6/11. Wrap up project 6/9 to allow for administrative close outs.

# GHG Decomposition

## Goal:
Decompose CO2 Emissions by fuel and sector (historical and projected)

## Required inputs:
- CO2 Emissions data (EIA):
    - Notes: 
        - EIA API contains endpoints for CO2 emissions by state (including total US), fuel and sector (including all)
        - Fuels (depend on sector?):
            - All fuels
            - Coal
            - Natural Gas
            - Petroleum
    - Link: https://www.eia.gov/opendata/qb.php?category=2251604 
- AEO (projections) includes Energy-Related Carbon Dioxide Emissions by Sector (including total), Source, and Region (including total US)
    - Sources (depend on sector):
        - Coal
        - Natural Gas
        - Petroleum
        - Electricity
        - Total
- CO2 Emissions factors (EPA): 
    - https://www.epa.gov/climateleadership/ghg-emission-factors-hub

## Framework (overall vision):
Multiply fuel use by emissions factor? (using constant factors over time)
Replacing energy use with emissions in decomp formula
plot difference between EIA emissions and calculated emissions

## Steps (specific action items):
- Determine a go/no-go decision point (on 3/12)
- Determine if emissions factors change over time (investigate using EIA emissions and energy, or EPA emissions factors)
- Determine existing fuel breakout and need for additional disaggregation (for decomposing fuel mix change) for each sector
- See if Ang has published a CO2 index decomposition; if so, use as a starting point (e.g., https://doi.org/10.1016/j.ecolecon.2013.06.007)
- Develop new methods and/or classes to disaggregate fuel mix appropriately (if need be) 
    - Develop associated tests
- Import collect_data methods from each sector specific class
- collect emissions factor data (from EIA or EPA data based on the fuel type) 
- multiply energy use data by emissions factor data, replace energy use data with the product (emissions by energy type)
- Write test comparing results to EIA emissions estimates (for both historical and projections)
- pass resulting data dictionaries to CalculateLMDI class

## Notes:
This flow follows similar process to EconomyWide, potentially inherit that class

## Desired outputs:
results of decomposition in csv and visualizations (as in the rest of EII)

## Timeline  with Deliverables:
Updated , well-documented  (push updated documentation to GitHub pages) code, with results data files and visualizations

## Timeline
2/26: Have inventory of existing fuel breakout, review relevant CO2 decomposition lit, determine if emissions factors change over time.
Revisit timeline based on existing fuel disaggregation (push out timeline if data aren't disaggregated)
4/2: develop required methds/classes and tests for importing new data, performing fuel disaggregations, and decomposing CO2 emissions
*4/30: finish testing and debugging. Final data, vizualizations, and documentation due [deliverable]*


# Manufacturing Sector Expansion

## Goal
Better capture structural changes in manufacturing sector for decomposition

## Required inputs:
Fuels and activity at the most disaggregate NAICS code level (based on MECS data)
SIC and NAICS code crosswalks (i.e., translation from SIC to NAICS 2002(7), to NAICS2012) [Colin has code to do this xwalk]

## Framework:
Use minimum and maximum NAICS digits for level of aggregation (e.g. minimum of 3 and maximum as high as MECS  allows)

## Steps:
- Determine a go/no-go decision point (on 3/12)
- Explore highest number of NAICS digits contained in each MECS year (differs by year); decide on which disaggregation is consistent across all MECS years.
- Fix ASM price fit module
- Change hard-coded data methods to import data from EIA website (read in and format Excel files)
- Estimate missing/withheld data (denoted by *, W, Q, etc.) [Colin has some code to do part of this process]
- Write test to check  results against sums
- Cross walk SIC/NAICS codes [Colin has code to do this]
- Find highest level of disaggregation for each 3-digit category; calculate missing "other " categories
- Write test to check aggregation against 3-digit sums
- Merge economic acitivty data based on matching NAICS codes, also calculate sum of activty data for missing "other" categories
- Write teststo check aggregation
- Document new methods/classes and update GitHub pages documentation
- Compare results to original 3-digit results

## Desired outputs:
- Standard EII outputs for the manufacturing sector (with binary levels of aggregation [3-digit or MECS maximum])
- Updated documentation
- Timeline  with Deliverables:


# Trade Flow Decomposition

## Goal:
Incorporate embodied energy from imported goods in decomposition of the Industrial Sector.
Account for offshoring bias in price deflators

## Required inputs:
BEA industry accounts data (trade extensions broken into intermediate/final goods)

## Framework:
Net out intermediate inputs imported

## Steps:
Lit review of any existing decomposition (focus on index, but also include structural decomposition work).


## Desired outputs:
results of decomposition in csv and visualizations (as in the rest of EII) where imports are a sub-sector level?

## Timeline  with Deliverables:
- 6/9 submit to Colin a near-final draft manuscript (colin to wrap up by 7/2). 

## Resources:
- [Offshoring Bias](https://www.federalreserve.gov/pubs/ifdp/2010/1007/ifdp1007.htm)
- [Measuring Globalization](https://research.upjohn.org/up_press/232/)
- [Intermediate Inputs and Economic Productivity](https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.2011.0565)
- [Industrial Energy Data Book](https://www.nrel.gov/docs/fy20osti/73901.pdf)