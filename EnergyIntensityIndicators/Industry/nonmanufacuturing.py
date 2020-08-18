
class NonManufacturing:
    """ Prior to 2012, total nonmanufacturing
    energy consumption (electricity and fuels) was estimated as a residual between the supply-side
    estimates of industrial consumption published by EIA and the end-user estimates for manufacturing
    based upon the MECS (supplemented by census-based data, as described above). The residual-based
    method produced very unsatisfactory results; year-to-year changes in energy consumption were
    implausible in a large number of instances. A complicating factor for fuels is that industrial consumption
    estimates published by EIA include energy products used as chemical feedstocks and other nonfuel
    purposes. As a result, a preliminary effort was undertaken in mid-2012 to estimate energy consumption
    from the user side for these sectors.   


    """    
    def __init__(self):
        pass

    def agriculture(self):
        miranowski_data =  # Annual Estimates of energy by fuel for the farm sector for the period 1965-2002
        nass_expenses_data =  # https://quickstats.nass.usda.gov/results/06763638-EB97-3879-AAF6-214CF147AED2

        nass_average_prices_data =  # 
        MER_fuel_price_data =  # 
        eia_table33 =  # Consumer Price estimates for Energy by Source, 1970-2009
        eia_table34 =  # Consumer price estimates for energy by end-use sector, 1970-2009
        eia_table523 = # All sellers sales prices for selected petroleum products, 1994-2010
        eia_table524 =  # Retail motor gasoline and on-highway diesel fuel prices, 1949-2010 
        
        pass

    def mining(self):
        """The energy consumption estimates for mining depend entirely on the various editions of the periodic
        census (ending in years with ‘2” and “7” since 1967). Up through 1987, the information for mining was
        collected under the title “Census of Mineral Industries.” From 1992 forward, the same information is
        part of the mining segment of the Economic Census (which now is the broad term for all the census
        surveys in the census years).
        Table A.10 shows the website data sources for the mining sector. For the most recent census in 2007,
        the data were selected from a flexible download procedure that allows the user to select key data
        elements for each specific NAICS sector. The specific data items were 1) “quantity of electricity
        purchased” and 2) fuels consumed by type: a) quantity, and b) delivered cost. For the previous years,
        the data were derived from downloaded industry series reports (or selected pages). In these reports,
        the cost and quantity of electricity is found in Table 3 (Detailed Statistics by Industry) and Table 7
        (Selected Supplies, Minerals Received for Preparation, Purchased Machinery, and Fuels Consumed by
        Type)

            Since 1997, the mining industries have been classified under three major 3-digit NAICS sectors: 211, Oil
        and Gas Extraction; 212, Mining (except oil and gas); 213 Support Activities for Mining. Unfortunately,
        there are no aggregations of energy data from the more detailed industries to this level. Thus, an
        estimation of electricity and fuel consumption must begin with the more detailed mining sectors,
        essentially 6-digit NAICS since 1997 and 4-digit SIC in earlier years. At the NAICS level, there are 29
        specific industries as shown in Table A.11 (the word “mining” has been omitted from most of the official
        NAICS titles in the table). 

        With regard to “Other fuels”, the assumption was that the dominant fuel was propane. The cost
        estimates were converted to quantities by the use of the price of propane published by EIA.1
        For
        undistributed fuels, the assumption was that the average price of the unreported fuels was the same as
        the reported fuels. Operationally, this assumption was implemented as follows. The cost and quantity
        of reported fuels was estimated. Then the ratio of the total cost of all fuels with respect to the cost of
        reported fuels was calculated. This ratio (> 1.0) was then used as multiplicative adjustment factor
        applied to the quantity of all reported fuels. 
        """        
        mining_2017 = 'https://www.census.gov/data/tables/2017/econ/economic-census/naics-sector-21.html'
        mining_2012 = 'https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?src=bkmk'
        mining_2007 = 'http://factfinder2.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ECN_2007_US_21SG12&prodType=table'
        mining_2002 = 'https://www.census.gov/econ/census02/guide/INDRPT21.HTM'  # extract Table 3 and Table 7
        mining_1997 = 'http://www.census.gov/prod/www/abs/ec1997mining-ind.html'  # extract Table 3 and Table 7
        mining_1992 = 'http://www.census.gov/prod/1/manmin/92mmi/92minif.html'   # extract Table 3 and Table 7
        

    def construction(self):
        pass