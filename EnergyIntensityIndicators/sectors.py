sectors = {'residential': {'Northeast': {'Single-Family': None, 'Multi-Family': None, 'Manufactured Homes': None}, 
                               'Midwest': {'Single-Family': None, 'Multi-Family': None, 'Manufactured Homes': None},
                               'South': {'Single-Family': None, 'Multi-Family': None, 'Manufactured Homes': None},
                               'West': {'Single-Family': None, 'Multi-Family': None, 'Manufactured Homes': None}},
              'industrial': {'Manufacturing': {'Food, Beverages, & Tobacco': None, 'Textile Mills and Products': None, 
                                               'Apparel & Leather': None, 'Wood Products': None, 'Paper': None,
                                               'Printing & Allied Support': None, 'Petroleum & Coal Products': None, 'Chemicals': None,
                                               'Plastics & Rubber Products': None, 'Nonmetallic Mineral Products': None, 'Primary Metals': None,
                                               'Fabricated Metal Products': None, 'Machinery': None, 'Computer & Electronic Products': None,
                                               'Electical Equip. & Appliances': None, 'Transportation Equipment': None,
                                               'Furniture & Related Products': None, 'Miscellaneous': None},
                             'Nonmanufacturing': {'Agriculture, Forestry & Fishing': None,
                                                  'Mining': {'Petroleum and Natural Gas': None, 
                                                             'Other Mining': None, 
                                                             'Petroleum drilling and Mining Services': None},
                                                  'Construction': None}}, 
              'commercial': {'Commercial_Total': None, 'Total_Commercial_LMDI_UtilAdj': None}, 
              'transportation': {'All_Passenger':
                                    {'Highway': 
                                        {'Passenger Cars and Trucks': 
                                            {'Passenger Car – SWB Vehicles': 
                                                {'Passenger Car': None, 'SWB Vehicles': None},
                                             'Light Trucks – LWB Vehicles': 
                                                {'Light Trucks': None, 'LWB Vehicles': None},
                                             'Motorcycles': None}, 
                                        'Buses': 
                                            {'Urban Bus': None, 'Intercity Bus': None, 'School Bus': None}, 
                                        'Paratransit':
                                            None}, 
                                    'Rail': 
                                        {'Urban Rail': 
                                            {'Commuter Rail': None, 'Heavy Rail': None, 'Light Rail': None}, 
                                        'Intercity Rail': None}, 
                                    'Air': {'Commercial Carriers': None, 'General Aviation': None}}, 
                                'All_Freight': 
                                    {'Highway': 
                                        {'Freight-Trucks': 
                                            {'Single-Unit Truck': None, 'Combination Truck': None}}, 
                                    'Rail': None, 
                                    'Air': None, 
                                    'Waterborne': None,
                                    'Pipeline': 
                                        {'Oil Pipeline': None, 'Natural Gas Pipeline': None}}}, 
              'electricity': {'Elec Generation Total': 
                                {'Elec Power Sector': 
                                    {'Electricity Only':
                                        {'Fossil Fuels': 
                                            {'Coal': None, 'Petroleum': None, 'Natural Gas': None, 'Other Gasses': None},
                                         'Nuclear': None, 
                                         'Hydro Electric': None, 
                                         'Renewable':
                                            {'Wood': None, 'Waste': None, 'Geothermal': None, 'Solar': None, 'Wind': None}},
                                     'Combined Heat & Power': 
                                        {'Fossil Fuels'
                                            {'Coal': None, 'Petroleum': None, 'Natural Gas': None, 'Other Gasses': None},
                                         'Renewable':
                                            {'Wood': None, 'Waste': None}}}, 
                                'Commercial Sector': None, 
                                'Industrial Sector': None},
                              'All CHP':
                                {'Elec Power Sector': 
                                    {'Combined Heat & Power':
                                        {'Fossil Fuels':
                                            {'Coal': None, 'Petroleum': None, 'Natural Gas': None, 'Other Gasses': None},
                                        'Renewable':
                                            {'Wood': None, 'Waste': None},
                                        'Other': None}},
                                    
                                'Commercial Sector':
                                    {'Combined Heat & Power':
                                        {'Fossil Fuels':
                                            {'Coal', 'Petroleum', 'Natural Gas', 'Other Gasses'},
                                        'Hydroelectric',
                                        'Renewable':
                                            {'Wood', 'Waste'},
                                        'Other'}},, 
                                'Industrial Sector':
                                    {'Combined Heat & Power':
                                        {'Fossil Fuels':
                                            {'Coal', 'Petroleum', 'Natural Gas', 'Other Gasses'},
                                        'Hydroelectric',
                                        'Renewable':
                                            {'Wood', 'Waste'},
                                        'Other'}}}}}