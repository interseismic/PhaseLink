# PhaseLink: A deep learning approach to seismic phase association

This code accompanies the paper
```
Ross, Z. E., Yue, Y., Meier, M.‐A., Hauksson, E., & Heaton, T. H. ( 2019). PhaseLink: A deep learning approach to seismic phase association. Journal of Geophysical Research: Solid Earth, 124, 856– 869. https://doi.org/10.1029/2018JB016674 [arXiv:1809.02880]
```

Please see the PDF user manual first: GPD_Phaselink_guide_20210331.pdf

Quick overview of codes:
There are four scripts that should be used in the following order:
1) phaselink_dataset.py : Build a training dataset from a station file and 1D travel time table. The travel time tables are in the format that is output by the GrowClust code. A decoupled version of this raytracer is provided in raytracer.tar.gz, which has a python wrapper to the F90 routine.

2) phaselink_train.py : Train a stacked bidirectional GRU model to link phases together. This code takes the training dataset produced in step 1 and trains a deep neural net to link together phase detections.

3) phaselink_eval.py : Associate a set of phase detections to earthquakes. This code runs the PhaseLink algorithm in evaluation mode, by using the trained model to link together phases and clustering the links to detect events. It outputs detections in NonLinLoc phase format to be located.

4) phaselink_plot.py : Plot resulting detections after locating them

More details about these codes and input file formats will be added over time. All of the scripts take a json filename as a command line argument. See the example file params.json.

Contact Zachary Ross (Caltech) with any questions.
