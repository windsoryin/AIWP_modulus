import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': 'land_sea_mask',
        'year': '2014',
        'month': '01',
        'day': '01',
        'time': '00:00',
    },
    'land_sea_mask.nc')