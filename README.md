# DataConsistencyConditions

This project describes the computation of fan beam dcc to cone beam dcc with cylindrical detector or flat detector, and any kind of trajectory. 

A short description of the different files follows:
  - GeneralFunctions.py: functions to extract projections from a stack, and it corresponding geometry.
  - FanbeamDccOnPhysicalDetector.py: function used to compute dcc for cone beam and cylindrical detector using a backprojection plane
  - FanbeamDccWithBackprojectionPlane.py: functions used to compute dcc directly on physical detectors, to draw geometrical illustrations
  - OnePairExample.ipynb: illustration of the moment computation for a pair of projections and that contains geometrical illustrations after
