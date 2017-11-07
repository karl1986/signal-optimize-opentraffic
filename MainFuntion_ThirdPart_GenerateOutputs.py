import glob
import os
import csv
import zipfile
import io
import sys
import numpy as np
from datetime import datetime, timedelta,time
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import matplotlib
import matplotlib.pyplot as plt
import time as timeSys

from joblib import Parallel, delayed  
import multiprocessing

#Change the path
os.getcwd()
os.chdir('E:/Projects/Liang Tang/WorldBank')
namelist=glob.glob('E:/Projects/Liang Tang/WorldBank/Data/Cebu_March_FirstWeek_Workday/grabtaxi-cebu-selected/*.csv')
pathname = 'E:/Projects/Liang Tang/WorldBank/Data/Cebu_March_FirstWeek_Workday'
directoryPath = 'E:/Projects/Liang Tang/WorldBank/PythonCode'

# Read in Functions file
sys.path.append(directoryPath)
import FastFunctions as Functions
reload(Functions)
haversine = Functions.haversine
filenameFilter = Functions.filenameFilter
try_parsing_date = Functions.try_parsing_date
try_parsing_dateTime = Functions.try_parsing_dateTime
try_parsing_time = Functions.try_parsing_time
weekdayFilter = Functions.weekdayFilter
hourFilter = Functions.hourFilter
latlongFilter = Functions.latlongFilter
readDFfromTxt = Functions.readDFfromTxt
addEndPointLabel = Functions.addEndPointLabel
findRegion = Functions.findRegion
addRegionTags = Functions.addRegionTags
keepEffectiveTrips = Functions.keepEffectiveTrips
mainFunction_new = Functions.mainFunction_new
plotStats = Functions.plotStats
cycleLengthAnalysis = Functions.cycleLengthAnalysis
mainFunction_step1_dataFilter = timeCalculate = Functions.timeCalculate
addDelay = Functions.addDelay
findTripFromGPSFile = Functions.findTripFromGPSFile
mainFunction = Functions.mainFunction
Functions.mainFunction_step1_dataFilter
mainFunction_step2_createTrips = Functions.mainFunction_step2_createTrips
mainFunction_step3_plotStats = Functions.mainFunction_step3_plotStats
mainFunction_step3_plotStats_SaveInOneFolder = Functions.mainFunction_step3_plotStats_SaveInOneFolder
mainFunction_step1_dataFilter_cebuNew = Functions.mainFunction_step1_dataFilter_cebuNew
mainFunction_PlotDelayCDF = Functions.mainFunction_PlotDelayCDF
signalTiming = Functions.signalTiming
mainFunction_step2_createTrips_cebuNew=Functions.mainFunction_step2_createTrips_cebuNew
mainFunction_step3_plotStats_SaveInOneFolder_eachSingleDay = Functions.mainFunction_step3_plotStats_SaveInOneFolder_eachSingleDay
signalTiming_eachSingleDay = Functions.signalTiming_eachSingleDay
peakHourCheckMinGroupPM = Functions.peakHourCheckMinGroupPM
removeNotPeakForSingleDay = Functions.removeNotPeakForSingleDay
mainFunction_step3_plotStats_SaveInOneFolder_mergeMultipleDays = Functions.mainFunction_step3_plotStats_SaveInOneFolder_mergeMultipleDays
mergeTripsFromMultiDays = Functions.mergeTripsFromMultiDays
signalTiming_AMPM = Functions.signalTiming_AMPM
signalTiming_EachHour = Functions.signalTiming_EachHour
#Define time difference threshold for trip building
timeDiffThres = timedelta(minutes=1)


intersectionDF = pd.read_csv('intersection_information.csv',index_col=0)


##Run the main function Step 3
start_time = timeSys.time()
for i in intersectionDF.index.values:
    if i != 36:
        tempList = list(intersectionDF.loc[i,])
        tempList1 = tempList[1:]
        mainFunction_step3_plotStats_SaveInOneFolder(namelist,pathname,timeDiffThres,*tempList1)

signalTiming_AMPM(pathname)
signalTiming_EachHour(pathname)
print("--- %s seconds ---" % (timeSys.time() - start_time))