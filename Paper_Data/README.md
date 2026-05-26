# Paper Data

This folder contains a 1000 sample segment of the processed DOG/GPS data used in the "Geodetic positioning on the deep seafloor using one-way travel-time ranging to continuously operating reference stations" paper. Upon publication the entire dataset will be posted here. If full data is desired now, please reach out and request from sk8609@princeton.edu

## File

```text
Processed_GPS_Receivers_DOG_3_sample.npz
```

## Contents

The file contains three NumPy arrays:

| Key | Shape | Description |
|---|---:|---|
| `GPS_Coordinates` | `(1000, 4, 3)` | ECEF coordinates of the four GPS antennas. |
| `GPS_data` | `(1000,)` | Processed GPS timing data. |
| `CDOG_data` | `(1000, 2)` | Processed DOG acoustic timing data. |

## Loading the data

```python
import numpy as np

data = np.load("Paper_Data/Processed_GPS_Receivers_DOG_3_sample.npz")

GPS_Coordinates = data["GPS_Coordinates"]
GPS_data = data["GPS_data"]
CDOG_data = data["CDOG_data"]
```
