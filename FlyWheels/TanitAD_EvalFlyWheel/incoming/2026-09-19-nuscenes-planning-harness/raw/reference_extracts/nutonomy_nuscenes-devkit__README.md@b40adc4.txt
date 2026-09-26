# PUBLISHED-CODE extract - nutonomy/nuscenes-devkit @ b40adc467b919192899405d9b77871afee8efa07
# path: README.md
# git blob sha1 of the fetched file: f47c2ce889ce6d289d86c9630d0060e096d685de (14960 B); verified against the GitHub contents API: MATCH
# fetched 2026-09-19T13:46:02Z from raw.githubusercontent.com (read-only source text; excerpts only, attribution retained)

## lines 94-110: setup: account + ToU + layout
   94  ## nuScenes
   95  
   96  ### nuScenes setup
   97  To download nuScenes you need to go to the [Download page](https://www.nuscenes.org/download), 
   98  create an account and agree to the nuScenes [Terms of Use](https://www.nuscenes.org/terms-of-use).
   99  After logging in you will see multiple archives. 
  100  For the devkit to work you will need to download *all* archives.
  101  Please unpack the archives to the `/data/sets/nuscenes` folder \*without\* overwriting folders that occur in multiple archives.
  102  Eventually you should have the following folder structure:
  103  ```
  104  /data/sets/nuscenes
  105      samples	-	Sensor data for keyframes.
  106      sweeps	-	Sensor data for intermediate frames.
  107      maps	-	Folder for all map files: rasterized .png images and vectorized .json files.
  108      v1.0-*	-	JSON tables that include all the meta data and annotations. Each split (trainval, test, mini) is provided in a separate folder.
  109  ```
  110  If you want to use another folder, specify the `dataroot` parameter of the NuScenes class (see tutorial).

## lines 138-164: CAN bus + map expansion + map versions
  138  ### CAN bus expansion
  139  In February 2020 we published the CAN bus expansion.
  140  It contains low-level vehicle data about the vehicle route, IMU, pose, steering angle feedback, battery, brakes, gear position, signals, wheel speeds, throttle, torque, solar sensors, odometry and more.
  141  To install this expansion, please follow these steps:
  142  - Download the expansion from the [Download page](https://www.nuscenes.org/download),
  143  - Extract the can_bus folder to your nuScenes root directory (e.g. `/data/sets/nuscenes/can_bus`).
  144  - Get the latest version of the nuscenes-devkit.
  145  - If you already have a previous version of the devkit, update the pip requirements (see [details](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/installation.md)): `pip install -r setup/requirements.txt`
  146  - Get started with the [CAN bus readme](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/can_bus/README.md) or [tutorial](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/can_bus_tutorial.ipynb).
  147  
  148  ### Map expansion
  149  In July 2019 we published a map expansion with 11 semantic layers (crosswalk, sidewalk, traffic lights, stop lines, lanes, etc.).
  150  To install this expansion, please follow these steps:
  151  - Download the expansion from the [Download page](https://www.nuscenes.org/download),
  152  - Extract the contents (folders `basemap`, `expansion` and `prediction`) to your nuScenes `maps` folder.
  153  - Get the latest version of the nuscenes-devkit.
  154  - If you already have a previous version of the devkit, update the pip requirements (see [details](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/installation.md)): `pip install -r setup/requirements.txt`
  155  - Get started with the [map expansion tutorial](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/map_expansion_tutorial.ipynb).
  156  For more information, see the [map versions](#map-versions) below.
  157  
  158  ### Map versions
  159  Here we give a brief overview of the different map versions:
  160  - **v1.3**: Add BitMap class that supports new lidar basemap and legacy semantic prior map. Remove [one broken lane](https://github.com/nutonomy/nuscenes-devkit/issues/493).
  161  - **v1.2**: Expand devkit and maps to include arcline paths and lane connectivity for the prediction challenge.
  162  - **v1.1**: Resolved issues with ego poses being off the drivable surface.
  163  - **v1.0**: Initial map expansion release from July 2019. Supports 11 semantic layers.
  164  - **nuScenes v1.0**: Came with a bitmap for the semantic prior. All code is contained in nuscenes.py.

