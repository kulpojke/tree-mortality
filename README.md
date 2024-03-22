# tree-mortality
Code for estimating tree mortality using Lidar and NAIP imagery.  This is the code developed for my Master of Science thesis, [Automated Tree Mortality Detection Using Ubiquitously Available Public Data](https://digitalcommons.calpoly.edu/theses/2761).  The code here is rather rough.  A production ready version is in progress in the repository `kulpojke/dual-sensor-tree-mortality`.

## Trinity County Mortality Predictions Workflow
+ Fire history and climate data are compiled in `src/trinity_fire_history_and_climate.ipynb`.
+ Geomorphons of various scales are created in `src/trinity_geomorphon.ipynb`.
+ Model selection occurs in `src/mortality_classification.ipynb`.
+ Tests for differences in model performance across different regions of the study are are performed in `html_notebooks/test_geographic_outlier.html`.
+ Code for Chapter 3 of the thesis are found in `html_notebooks/helena_geomorphon.html`, `html_notebooks/helena_mortality_part1.html`, and `html_notebooks/helena_mortality_part2_5.html`.

