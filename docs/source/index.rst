.. EnergyIntensityIndicators documentation master file, created by
   sphinx-quickstart on Thu Jul 23 10:02:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EnergyIntensityIndicators's documentation!
======================================================

EnergyIntensityIndicators provides a framework to quantify how energy efficiency may or may not be affecting energy use relative to other economic trends. 

This model is based on decades of work by the Pacific Northwest National Laboratory (PNNL) for the Department of Energy's 
Office of Renewable Efficiency and Renewable Energy (EERE). The original PNNL `methodology <https://github.com/NREL/EnergyIntensityIndicators/blob/master/Original%20documentation/EII_Summary_Process_Flow_PNNL_29905.pdf>`_ 
documents these efforts, the results of which, along with related analyses are documented `here <https://www.energy.gov/eere/analysis/energy-intensity-indicators>`_.

The model enables the decomposition of energy use for economic sectors through the use of a Log Mean Divisia Index (LMDI) model. The LMDI methodology 
decomposes energy use into three main categories: activity, structure and intensity (i.e. energy efficiency). A user can thus explain changes to 
overall energy use in a sector through changes to sector output (activity), structural shifts within sectors (e.g. across transportation modes), and efficiency of 
energy use in that sector. 

The LMDI model, developed by B.W. Ang and K. Choi, has several permutations featuring varying mathematical characteristics. LMDI-I is "consistent in aggregation" and 
"perfect in decomposition at the subcategory level," while the LMDI-II model is preferred for index construction (Ang, 2015).  
In order to enable users to choose the LMDI version that best suits their needs, 
this EnergyIntensityIndicators library allows for choice between Additive (LMDI-I or LMDI-II) Multiplicative (LMDI-II) models. 
The LMDI model and its various versions are explained in further detail by Ang `here <https://doi.org/10.1016/j.enpol.2015.07.007>`_ , with additional 
materials linked in the Relevant Background Literature section below.

This project aims to add flexibility to the original model, which was implemented in Excel spreadsheets; 
the EII framework offers user choice in the following dimensions: 

- Additive (LMDI-I or LMDI-II) vs Multiplicative LMDI (LMDI-II)
- Base Year (default 1985)
- Economic Sector

The LMDI module outputs results in formatted csv for each sector and model appropriate visualizations (i.e. waterfall charts for additive, line charts for multiplicative) 
for each level of aggregation desired. 

===============================
Relevant Background Literature
===============================

- Belzer, D. B. (2014). `A Comprehensive System of Energy Intensity Indicators for the U.S.: Methods, Data and Key Trends <https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-22267.pdf>`_ (PNNL-22267). Pacific Northwest National Laboratory (PNNL). 
- Ang, B. W. (2005). `The LMDI approach to decomposition analysis: a practical guide <https://doi.org/10.1016/j.enpol.2003.10.010>`_. Energy Policy, 33(7), 867–871.  
- Ang, B. W. (2015). `LMDI decomposition approach: A guide for implementation <https://doi.org/10.1016/j.enpol.2015.07.007>`_. Energy Policy, 86, 233–238. 
- Su, B., & Ang, B. W. (2012). `Structural decomposition analysis applied to energy and emissions: Some methodological developments <https://doi.org/10.1016/j.eneco.2011.10.009>`_. Energy Economics, 34(1), 177–188.  
- Wang, H., Ang, B. W., & Su, B. (2017). `Multiplicative structural decomposition analysis of energy and emission intensities: Some methodological issues <https://doi.org/10.1016/j.energy.2017.01.141>`_. Energy, 123, 47–63.  
- Wang, H., Ang, B. W., & Su, B. (2017). `Assessing drivers of economy-wide energy use and emissions: IDA versus SDA. <https://doi.org/10.1016/j.enpol.2017.05.034>`_. Energy Policy, 107, 585–599.  

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   EnergyIntensityIndicators/LMDI/lmdi
   EnergyIntensityIndicators/lmdi_gen
   EnergyIntensityIndicators/Commercial/commercial
   EnergyIntensityIndicators/Electricity/electricity
   EnergyIntensityIndicators/Industry/industry
   EnergyIntensityIndicators/Residential/residential
   EnergyIntensityIndicators/Transportation/transportation
   EnergyIntensityIndicators/Emissions/emissions
   EnergyIntensityIndicators/supporting_files
   EnergyIntensityIndicators/utilities/utilities
   EnergyIntensityIndicators/results
  


Indices and tables
==================

* :ref:`modindex`
