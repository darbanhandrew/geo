import matplotlib.pyplot as plt
from rasterio.plot import show
import ee
import rasterio
import tensorflow as tf
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from numpy import savetxt
import simplekml
import numpy as np

ee.Authenticate()
ee.Initialize()


def main(start_time=None, end_time=None, geoJson=None, order_id=None):
    Time_zone = 'Etc/GMT-3';
    time_start = ee.Date('2019-3-22T00:00:00', Time_zone);
    time_end = time_start.advance(1, 'day', Time_zone)
    geoJSON = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [43.40228622016353, 24.238373094311648],
                        [64.05658309516353, 24.238373094311648],
                        [64.05658309516353, 40.33936254411409],
                        [43.40228622016353, 40.33936254411409],
                        [43.40228622016353, 24.238373094311648]
                    ]
                }
            }
        ]
    }
    coords = geoJSON['features'][0]['geometry']['coordinates']
    aoi = ee.Geometry.Polygon(coords)
    types = [
        ("COPERNICUS/S5P/NRTI/L3_AER_AI", 'absorbing_aerosol_index', 'aerosol'),
        ("COPERNICUS/S5P/NRTI/L3_CO", 'CO_column_number_density', 'co'),
        ("COPERNICUS/S5P/NRTI/L3_HCHO", 'tropospheric_HCHO_column_number_density', 'formaldehyde'),
        ("COPERNICUS/S5P/OFFL/L3_CH4", 'CH4_column_volume_mixing_ratio_dry_air', 'methane'),
        ('COPERNICUS/S5P/NRTI/L3_NO2', 'NO2_column_number_density', 'no2'),
        ("COPERNICUS/S5P/NRTI/L3_O3", 'O3_column_number_density', 'ozone'),
        ("COPERNICUS/S5P/NRTI/L3_SO2", 'SO2_column_number_density', 'so2')

    ]
    for (address, select, title) in types:
        sentinel_5p_NO2 = (ee.ImageCollection(address).
                           filterBounds(aoi).
                           filterDate(time_start, time_end).
                           select(select).
                           sort('system:time_start').
                           mean().
                           clip(aoi)
                           )
        scale = 10000
        url = sentinel_5p_NO2.getDownloadURL(
            params={'name': title, 'scale': scale, 'region': aoi, 'crs': 'EPSG:4326', 'filePerBand': False})

        zipurl = url
        with urlopen(zipurl) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('assets/')
        dataset = rasterio.open('assets/' + title + '.tif')
        dataset_array = dataset.read(1)
        savetxt('data.csv', dataset_array, delimiter=',')
        row, column = dataset_array.shape
        print(row, column)
        print(title)
        dataset_array_flat = dataset_array.flatten()
        new_model = tf.keras.models.load_model('my_model.h5')
        predictions = new_model.predict(dataset_array_flat)
        longitude_boundaries = [43.40228622016353, 64.05658309516353]
        latitude_boundaries = [40.33936254411409, 24.238373094311648]
        longitude_step = (longitude_boundaries[1] - longitude_boundaries[0]) / column
        latitude_step = (latitude_boundaries[0] - latitude_boundaries[1]) / row
        normalized = (predictions - min(predictions)) / (max(predictions) - min(predictions))
        kml = simplekml.Kml()
        for (index, _), normalize in np.ndenumerate(normalized):
            long = (index % column) * longitude_step + longitude_boundaries[0]
            lat = latitude_boundaries[0] - (int(index / column)) * latitude_step
            name = "Block " + str(index)
            value = normalize
            kml.newpoint(name=name, coords=[(long, lat)], description=str(value))

        kml.save("kml/" + title + ".kml")
        final_data = predictions.reshape(row, column)


# if __name__ == "__main__":
#     main()
