# PhaseLink: A deep learning approach to seismic phase association

This code accompanies the paper
```
Ross, Z. E., Yue, Y., Meier, M.‐A., Hauksson, E., & Heaton, T. H. ( 2019). PhaseLink: A deep learning approach to seismic phase association. Journal of Geophysical Research: Solid Earth, 124, 856– 869. https://doi.org/10.1029/2018JB016674 [arXiv:1809.02880]
```
There are four scripts that should be used in the following order:
1) phaselink_dataset.py : Build a training dataset from a station file
2) phaselink_train.py : Train a stacked bidirectional GRU model to link phases together
3) phaselink_eval.py : Associate a set of phase detections to earthquakes
4) phaselink_plot.py : Plot resulting detections after locating them

More details about these codes and input file formats will be added over time. All of the scripts take a json filename as a command line argument. See the example file gpd.json. phaselink_eval.py will output the detections and phases in a NonLinLoc format for easy locations.

Contact Zachary Ross (Caltech) with any questions.
