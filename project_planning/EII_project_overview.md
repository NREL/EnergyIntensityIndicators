# EEI Project
Energy intensity indicators provide a way to quantify how energy efficiency may or may not be affecting energy use relative to other economic trends. EERE’s current approach to calculating its energy intensity indicators is spreadsheet based and requires manual updating. Additionally, several critical assumptions and calculation steps are not included with the publicly-available spreadsheets. The indicators are based on historical data, but it is possible to provide forward-looking analysis of energy projections. Visualizing and interacting with the indicators is limited to static data and figures hosted on the [DOE website](https://www.energy.gov/eere/analysis/energy-intensity-indicators). Developing an interactive web visualization tool will create more opportunities to interact with the indicators, potentially increasing their value and widening their audience.

## Description
- **Task 1** will This effort will move the calculation of EERE historical energy intensity indicators from the current spreadsheet-based approach to an open-source programming language (i.e. Python). A publicly-available calculation code repository will be created on GitHub. The repository will also document the calculation methods and source data.
- **Task 2** will provide a complement to the historical indicators by calculating energy intensity indicators of projected energy use from the Energy Information Administration's (EIA) Annual Energy Outlook (AEO). This will allow EERE stakeholders to examine how energy efficiency may or may not play a role in energy projections developed by EIA.
- **Task 3** will develop an interactive web visualization that will allow users to view and explore the various components the historical and projected indicators. The visualization will be modeled on the decomposition tool developed by the European Union’s ODYSEE-MURE project.

### Use cases
* Run decomposition analysis from command line, specifying base year, LMDI approach,
and economic sector (all, individual, and combinations)
* Output formatted for upload to visualization project
* User specifies which Annual Energy Outlook to decompose (automatically decomposes latest AEO)

## Goals
* Enable automated data updates by translating existing EII calculations and data from manually-updated spreadsheets to
Python and API-based data collection.
* Bring transparency to off-line calculations (e.g., data interpolation calculated
outside of EII spreadsheets).
* Create decomposition of EIA Annual Energy Outlook projections
* Tie results into data visualization project (**Task 3**).

## Deliverables
- GitHub repository for energy intensity indicators code and associated documentation
- Formatted data set of final energy intensity indicators as NREL Data Catalog entry
- *Additional (?):* Consumption-based decomposition calculations
- *Additional (?):* paper linking existing and new decomposition approaches for U.S.

## Rough Project Plan

| Task  |  Duration (Completion Date)|Deliverable |
| ----- | --------- | ------- |
| Review PNNL Methodology and EIA AEO data | 2 weeks (6/26) | No|
| Create draft model architecture and data connections| 2 weeks (7/10)| No|
| Code data collection (API) | 2 weeks (7/24) | No |
| Model beta (historical and projection)| 7 weeks (9/4) | No |
| Draft model documentation (GitHub) | 7 weeks (9/4) | No|
| Model and documentation review | 2 weeks (9/18)|No |
| Model revisions | 3 weeks (10/9) |No |
| Final model and documentation | 1 week (10/16)|**Yes** |
| Final data uploaded to NREL data catalog | (10/30)| **Yes**|

### Miscellaneous Thoughts
* Model should offer choice in calculating LDMI (i.e., both additive and multiplicative form)
* The decomposition of EIA AEO projections may be specified using less detailed data than
the historical decomposition.

## Relevant Background Literature
**Belzer, D. B. (2014). A Comprehensive System of Energy Intensity Indicators for the U.S.: Methods, Data and Key Trends (PNNL-22267). Pacific Northwest National Laboratory (PNNL). https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-22267.pdf

Ang, B. W. (2015). LMDI decomposition approach: A guide for implementation. *Energy Policy*, 86, 233–238. https://doi.org/10.1016/j.enpol.2015.07.007

Ang, B. W. (2005). The LMDI approach to decomposition analysis: a practical guide. *Energy Policy*, 33(7), 867–871. https://doi.org/10.1016/j.enpol.2003.10.010

Ang, B. W. (2015). LMDI decomposition approach: A guide for implementation. *Energy Policy*, 86, 233–238. https://doi.org/10.1016/j.enpol.2015.07.007**

Su, B., & Ang, B. W. (2012). Structural decomposition analysis applied to energy and emissions: Some methodological developments. *Energy Economics, 34(1)*, 177–188. https://doi.org/10.1016/j.eneco.2011.10.009

Wang, H., Ang, B. W., & Su, B. (2017). Multiplicative structural decomposition analysis of energy and emission intensities: Some methodological issues. *Energy*, 123, 47–63. https://doi.org/10.1016/j.energy.2017.01.141

Wang, H., Ang, B. W., & Su, B. (2017). Assessing drivers of economy-wide energy use and emissions: IDA versus SDA. *Energy Policy*, 107, 585–599. https://doi.org/10.1016/j.enpol.2017.05.034

**Velasco-Fernández, R., Dunlop, T., & Giampietro, M. (2019). Fallacies of energy efficiency indicators: Recognizing the complexity of the metabolic pattern of the economy. *Energy Policy*, 111089. https://doi.org/10.1016/j.enpol.2019.111089**

## Project Resources
* [GitHub repo](https://github.com/NREL/EnergyIntensityIndicators/)
* [Revised PNNL methodology and data](https://github.com/NREL/EnergyIntensityIndicators/tree/master/Original%20documentation)
* [EERE EEI website](https://www.energy.gov/eere/analysis/energy-intensity-indicators)
* [EIA Annual Energy Outlook](https://www.eia.gov/outlooks/aeo/)
* [Odyssee Database](https://www.indicators.odyssee-mure.eu/energy-efficiency-database.html)
* [Odyssee Decomposition Tool](https://www.indicators.odyssee-mure.eu/decomposition.html)
