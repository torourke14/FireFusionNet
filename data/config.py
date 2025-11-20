# Who wants to deal with tuples in JSON, anyways??
def get_config():
    return {
        "CDS": {
            "latlon": [49.9, -125.0, 45.0, -114.0],
            "dataset": "satellite-land-cover",
            "max_parallel_req": 5,
            "years": list([
                "2000", "2001", "2002", "2003", "2004",
                "2005", "2006", "2007", "2008", "2009",
                "2010", "2011", "2012", "2013", "2014",
                "2015", "2016", "2017",
                "2018", "2019", "2020"
            ])
        },
        "MODIS": {
            "latlon": tuple(-125.0, 45.0, -114.0, 49.9),
            "max_parallel_req": 7,
            # MODIS short names and versions
            "products": {
                "MYD11A1": "061",   # Land Surface Temperature/Emissivity
                "MOD13Q1": "061",   # Vegetation indices
                "MCD15A2H": "061",  # Leaf Area Index (LAI)
                "MCD64A1": "061"   # Burned area
            },
            "years": list([
                "2000", "2001", "2002",
                "2003", "2004", "2005", 
                "2006", "2007", "2008", 
                "2009", "2010", "2011", 
                "2012", "2013", "2014",
                "2015", "2016", "2017",
                "2018", "2019", "2020"
            ]),
            # Granule HDF file names are formatted as "<poduct>/<year>/<day-of-year>/<granule-files>.hdf
            # ex: MOD13Q1.A2000049.h09v05.006.2015136104623.hdf
            # equals: MOD13Q1, year 2000, day 49, h09/v05, hash
            # We only want tiles that overlay the Pacific Northwest, which covers the below h/v indices
            "tiles": ["h08v04", "h08v05", "h09v04", "h09v05", "h10v04"],
            # Specific parameter to extract
            "param_map": {
                "MOD13Q1": "250m 16 days NDVI",
                "MYD11A1": "LST_Day_1km", 
                "MCD15A2H": "Lai_500m",
                "MCD64A1": "Burn Date",
            },
        }
    }