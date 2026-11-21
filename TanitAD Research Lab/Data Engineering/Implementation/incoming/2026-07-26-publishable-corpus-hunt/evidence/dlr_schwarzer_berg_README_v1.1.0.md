# OpenDRIVE road network dataset of Schwarzer Berg in Brunswick

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15395839.svg)](https://doi.org/10.5281/zenodo.15395839) [![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This lane-detailed HD map as OpenDRIVE dataset models road network parts of the "Schwarzer Berg" region in the city of Brunswick, Lower Saxony, Germany. The main application scopes of this data are in the domain of automated driving: simulation, verification and validation.

![Surveyed road data](bs-schwarzer-berg.png?raw=true "Surveyed road data")

## Dataset origin and key features

The raw data was surveyed through mobile mapping in the beginning of 2024 and has not been updated since. From this raw data, road infrastructure elements have been modelled in accordance with the [ASAM OpenDRIVE 1.6](https://www.asam.net/standards/detail/opendrive/) specification. The OpenDRIVE dataset is a prototypic and simplified modelling of road elements and is intended to be used in research and development only. Neither the completeness nor the correctness of the data can be guaranteed in any way. The dataset author does not take legal responsibility for the use of this dataset.

The absolute coordinate error is expected to be less than 20 cm for road elements within the drivable surface (e.g. driving lane boundaries, road marks) with possibly greater absolute positioning error for elements with increasing distance from the road reference line.

The OpenDRIVE model encompasses:

- elevation profile and lateral profile with shape definition
- lanes
  - bicycle and sidewalk lanes – which are crossing regular driving lanes – are only modelled for visual completeness, without logical routing capability
  - tram lanes are only included for visual completeness, without logical routing capability
- road markings
- traffic signals and signs
- different other road infrastructure and road furniture objects 
- building facades

## GIS data

The GeoPackage file `bs-schwarzer-berg.gpkg.zip` can be used as complementary dataset in any common GIS software. It has been generated with `ogr2ogr` using the [XODR driver](https://gdal.org/en/stable/drivers/vector/xodr.html) of GDAL 3.10 with the following options:

- `DISSOLVE_TIN=NO`
- `EPSILON=0.1`

## Spatial coordinate reference system

File `bs-schwarzer-berg.xodr` is spatially referenced in [ETRS89/DREF91/2016 / UTM zone 32N](https://spatialreference.org/ref/epsg/10732/) with elevation information as [EGM2008 height](https://spatialreference.org/ref/epsg/3855/). The well-known text coordinate reference system definition is provided in file `ETRS89_DREF91_2016_UTM_zone_32N_EGM2008_height.wkt`.

Additionally, file `bs-schwarzer-berg_offset.xodr` defines a planar coordinate offset in `x` and `y` in order to avoid floating point inaccuracies in primitive visualisation applications:

- `x_offset = 601000.0`
- `y_offset = 5787000.0`

This offset is expressed as UTM false easting of `-101000` and false northing of `-5787000` in the corresponding PROJ [`tmerc`](https://proj.org/en/stable/operations/projections/tmerc.html) projection definition in the `<geoReference>` tag.

## Citation

If you use this dataset, please cite the specific version using its DOI and the metadata provided in the [CITATION file](CITATION.cff). For more information on the format, please see [Citation File Format (CFF)](https://citation-file-format.github.io/).

## Licence

This HD road network dataset is licenced under the terms of [Creative Commons Attribution 4.0 International](LICENSE.txt). Example for a licence attribution string:

> OpenDRIVE dataset Schwarzer Berg © DLR and iMAR Navigation GmbH, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), 2024
