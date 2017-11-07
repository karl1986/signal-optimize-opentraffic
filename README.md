# signal-optimize-opentraffic
---
The python code can create the by-turn traffic signal performance metric based on the GPS data obtained through the Open Traffic platform collaborated between the World Bank and Grab.
Upon the creation of the metric, signal timing plans could be adjusted iteratively to improve the mobility performance (i.e., delay).
Detailed algorithm of creating the metric and adjusting the signal plans could be found in the attached TRR paper.

## Introduction to the python code
---
There are four python code file:
1. FastFunctions.py includes all the functions, while other three python files call the functions in FastFunctions step by step to produce outputs
2. MainFuntion_FirstPart_FilterDataInTheIntersectionArea.py is the first step which filters the data and save the data records within the studied are
    Input:
        * intersection_information.csv which contains the geospatial information of all the studied intersections, as well as the speed limit of intersected corridors. These intersection information are read in as variable intersectionDF
        * All the GPS data files. These names of these data files are stored in namelist
    Output:
        * Filtered text files within the study area
3. MainFuntion_SecondPart_ConstructTrips.py is the second step which reads the output from the first step, identify trips from the GPS data which may be right turn, left turn or through trip on different intersection legs.
    Input:
        * intersection_information.csv which contains the geospatial information of all the studied intersections, as well as the speed limit of intersected corridors. These intersection information are read in as variable intersectionDF
        * Output from the previous step, which are the filtered text files within the study area
    Output:
        * Trips saved in different text files based on turns on different intersection legs
4. MainFuntion_ThirdPart_GenerateOutputs.py is the third and last step which calculate calculates statistics for different time of day and different directions and generates plots
    Input:
        * intersection_information.csv which contains the geospatial information of all the studied intersections, as well as the speed limit of intersected corridors. These intersection information are read in as variable intersectionDF
        * Output from the previous step, which are the trips saved in different text files based on turns on different intersection legs
    Output:
        * Calculate statistics like the mean value of delay and queuePercentageLength for different time of day and different directions. Results are saved in various csv files and plot files.

## How to use the python code
---
1. Prepare the input files:
    * intersection_information.csv which contains the geospatial information of all the studied intersections, as well as the speed limit of intersected corridors. These intersection information are read in as variable intersectionDF
    * All the GPS data files. These names of these data files are stored in namelist
2. Change the paths in the python files to the local paths
3. Run MainFuntion_FirstPart_FilterDataInTheIntersectionArea.py
4. Run MainFuntion_SecondPart_ConstructTrips.py
5. Run MainFuntion_ThirdPart_GenerateOutputs.py

## Authors
* [Liang Tang](liang@umd.edu)
* [Yang Carl Lu](ylu2@worldbank.org)