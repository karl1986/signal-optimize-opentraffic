import glob
import os
import csv
import zipfile
import io
import sys
## when use in the server
#packagepath = 'C:/Anaconda/Lib/site-packages'
#sys.path.append(packagepath)
import numpy as np
from datetime import datetime, timedelta,time, date
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
from joblib import Parallel, delayed  
import multiprocessing
import gzip
import shutil, errno

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Unit: m

    Parameters:
    -------------
    lat1, lon1: lat long of the first point
    lat2, lon2: lat long of the second point
    
    Returns:
    -------------
    haversine distance between the two points, with unit of meter
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000
    
def filenameFilter(filename):
    '''
    Based on the file name, determine if we want to analyze the data or not
    keep hours 7,8,9,10,15,16,17,18
    keep weekdays

    Parameters:
    -------------
    filename: the name of the file, string
    
    Returns:
    -------------
    flag: an int. 0: not in peak hours in weekday; 1: in peak hours in weekday; 2: Can't determine
    '''
    filename1 = filename.split('.')[0]
    filenameSplit = filename1.split('-')
    flag = 0
    if len(filenameSplit) == 3:
        tempDateString = filenameSplit[1] + '_' + filenameSplit[2]
        tempDate= datetime.strptime(tempDateString, '%Y%m%d_%H%M')
        if tempDate.weekday()<5:
            if (not (tempDate.hour == 7 and tempDate.minute == 0)) and (not (tempDate.hour == 15 and tempDate.minute == 0)):
                if (7<=tempDate.hour<=10 or 15<=tempDate.hour<=18) or  (tempDate.hour==11 and tempDate.minute == 0) or (tempDate.hour == 19 and tempDate.minute == 0):
                    flag = 1
    else:
        flag = 2
    return flag

def filenameWeekFilter(filename):
    '''
    Based on the file name, determine if we want to analyze the data or not
    keep hours 7,8,9,10,15,16,17,18
    keep weekdays

    Parameters:
    -------------
    filename: the name of the file, string
    
    Returns:
    -------------
    flag: an int. 0: not in peak hours in weekday; 1: in peak hours in weekday; 2: Can't determine
    '''
    filename1 = filename.split('.')[0]
    filenameSplit = filename1.split('-')
    flag = 0
    if len(filenameSplit) == 3:
        tempDateString = filenameSplit[1] + '_' + filenameSplit[2]
        tempDate= datetime.strptime(tempDateString, '%Y%m%d_%H%M')
        if tempDate.weekday()<5:
            flag = 1
    else:
        flag = 2
    return flag

def filenameWeekFilter_new(filename):
    '''
    Based on the file name, determine if we want to analyze the data or not
    keep hours 7,8,9,10,15,16,17,18
    keep weekdays

    Parameters:
    -------------
    filename: the name of the file, string
    
    Returns:
    -------------
    flag: an int. 0: not in peak hours in weekday; 1: in peak hours in weekday; 2: Can't determine
    '''
    filename1 = filename.split('.')[0]
    filenameSplit = filename1.split('_')
    flag = 0
    if len(filenameSplit) > 3:
        tempDate = date(int(filenameSplit[0]),int(filenameSplit[1]),int(filenameSplit[2]))
        if tempDate.weekday()<5:
            flag = 1
    return flag

def try_parsing_date(text):
    '''
    Convert a '%Y-%m-%d' type string to datetime object
    Parameters:
    -------------
    text: date string
    
    Returns:
    -------------
    datetime: datetime object
    '''
    fmt = '%Y-%m-%d'
    return datetime.strptime(text, fmt)
    #for fmt in ('%Y-%m-%d'):
    #    try:
    #        return datetime.strptime(text, fmt)
    #    except ValueError:
    #        pass
    #raise ValueError('no valid date format found')

def try_parsing_dateTime(text):
    '''
    Convert a '%Y-%m-%d %H:%M:%S' type string to datetime object
    Parameters:
    -------------
    text: datetime string
    
    Returns:
    -------------
    datetime: datetime object
    '''
    fmt = '%Y-%m-%d %H:%M:%S'
    return datetime.strptime(text, fmt)

def try_parsing_time(text):
    '''
    Convert a '%H:%M:%S' type string to datetime object
    Parameters:
    -------------
    text: time string
    
    Returns:
    -------------
    datetime: datetime object
    '''
    fmt = '%H:%M:%S'
    return datetime.strptime(text, fmt)
    
def weekdayFilter(dateString):
    '''
    Determine if a date type String is a weekday or not
    Parameters:
    -------------
    dateString: date type string
    
    Returns:
    -------------
    Boolean: True if it's a weekday, False if not
    '''
    dateT = try_parsing_date(dateString)
    if dateT.weekday() < 5:
        return True
    else:
        return False
def hourFilter(hour):
    '''
    Determine if an hour is a peak hour or not
    Parameters:
    -------------
    hour: int
    
    Returns:
    -------------
    Boolean: True if it's a peak hour, False if not
    '''
    if 7<=hour<=10 or 15<=hour<=18:
        return True
    else:
        return False

def latlongFilter(lat,lon,latN,latS,longE,longW):
    '''
    Filter data:
        keep data within the intersection square area
    Parameters:
    -------------
    lat,lon: lat long of the point
    latN,latS,longE,longW: lat long of the square area
    
    Returns:
    -------------
    Boolean: True if it's within the square area, False if not
    '''
    latMin = min(latN,latS)
    latMax = max(latN,latS)
    longMin = min(longE,longW)
    longMax = max(longE,longW)
    if longMin <= lon <= longMax and latMin<= lat <= latMax:
        return True
    else:
        return False

def dataFilterMainAllDay(namelist,folderFullName,latN,latS,longE,longW):
    '''
    Read data from namelist, keep data for the whole day within a rectangle region and save data to outfilename
    Parameters:
    -------------
    namelist: namelist including all files to be read in
    outfilename: output file name
    latN,latS,longE,longW: lat long of the square area
    
    Returns:
    -------------
    None. Filtered data is saved in outfilename
    '''
    dataFolderName = folderFullName + '/' + 'filteredData'
    if not os.path.isdir(dataFolderName):
        os.makedirs(dataFolderName)
    for name in namelist:
        base = os.path.basename(name)
        filename = os.path.splitext(base)[0]
        flag = filenameWeekFilter(filename)
        if flag == 1:
            tempFolderName = filename.split('-')[1]
            dataFolderFullName = dataFolderName + '/' + tempFolderName
            if not os.path.isdir(dataFolderFullName):
                os.makedirs(dataFolderFullName)
            csv_file = filename #all fixed
            zfile = zipfile.ZipFile(name)
            datafile = zfile.open(csv_file,'r') #don't forget this line!
            datafile1=io.TextIOWrapper(datafile, encoding='utf-8', newline='')
            reader = csv.reader(datafile1)
            i=0
            for line in reader:
                data=line
                dateframe=data[0].split()
                timeframe=dateframe[1].split('.')
                if len(timeframe) == 1:
                    timeframe = timeframe[0].split('+')
                if latlongFilter(float(data[3]),float(data[2]),latN,latS,longE,longW):
                    writeFileName = dataFolderFullName + '/' + data[1] + '.txt'
                    testfile=open(writeFileName,'a')
                    testfile.write(dateframe[0]+','+timeframe[0]+','+data[1]+','+data[3]+','+data[2]+',' + data[6] + '\n')	
                    testfile.close		
                i=i+1
        elif flag == 2:
            csv_file = filename #all fixed
            zfile = zipfile.ZipFile(name)
            datafile = zfile.open(csv_file,'r') #don't forget this line!
            datafile1=io.TextIOWrapper(datafile, encoding='utf-8', newline='')
            reader = csv.reader(datafile1)
            i=0
            for line in reader:
                data=line
                dateframe=data[0].split()
                timeframe=dateframe[1].split('.')
                if len(timeframe) == 1:
                    timeframe = timeframe[0].split('+')
                if weekdayFilter(dateframe[0]) and latlongFilter(float(data[3]),float(data[2]),latN,latS,longE,longW):
                    tempFolderName = dateframe[0]
                    dataFolderFullName = dataFolderName + '/' + tempFolderName
                    if not os.path.isdir(dataFolderFullName):
                        os.makedirs(dataFolderFullName)
                    writeFileName = dataFolderFullName + '/' + data[1] + '.txt'
                    testfile=open(writeFileName,'a')
                    testfile.write(dateframe[0]+','+timeframe[0]+','+data[1]+','+data[3]+','+data[2]+',' + data[6] + '\n')			
                i=i+1
    return

def dataFilterMainAllDay_cebuNew(namelist,folderFullName,latN,latS,longE,longW):
    '''
    Read data from namelist, keep data for the whole day within a rectangle region and save data to outfilename
    Parameters:
    -------------
    namelist: namelist including all files to be read in
    outfilename: output file name
    latN,latS,longE,longW: lat long of the square area
    
    Returns:
    -------------
    None. Filtered data is saved in outfilename
    '''
    dataFolderName = folderFullName + '/' + 'filteredData'
    if not os.path.isdir(dataFolderName):
        os.makedirs(dataFolderName)
    for name in namelist:
        base = os.path.basename(name)
        filename = os.path.splitext(base)[0]
        tempFolderName = filename.split('-')[0]
        dataFolderFullName = dataFolderName + '/' + tempFolderName
        if not os.path.isdir(dataFolderFullName):
            os.makedirs(dataFolderFullName)
        datafile1 = open(name, 'r')
        reader = csv.reader(datafile1)
        i=0
        for line in reader:
            data=line
            dateframe=data[0].split()
            if latlongFilter(float(data[9]),float(data[10]),latN,latS,longE,longW):
                writeFileName = dataFolderFullName + '/' + data[1] + '.txt'
                testfile=open(writeFileName,'a')
                testfile.write(dateframe[0]+','+dateframe[1]+','+data[1]+','+data[9]+','+data[10]+',' + data[8] + '\n')	
                testfile.close		
            i=i+1
    return

def dataFilterMainAllDay_new(namelist,folderFullName,latN,latS,longE,longW):
    '''
    Read data from namelist, keep data for the whole day within a rectangle region and save data to outfilename
    Parameters:
    -------------
    namelist: namelist including all files to be read in
    outfilename: output file name
    latN,latS,longE,longW: lat long of the square area
    
    Returns:
    -------------
    None. Filtered data is saved in outfilename
    '''
    dataFolderName = folderFullName + '/' + 'filteredData'
    if not os.path.isdir(dataFolderName):
        os.makedirs(dataFolderName)
    for name in namelist:
        base = os.path.basename(name)
        filename = os.path.splitext(base)[0]
        flag = filenameWeekFilter_new(filename)
        if flag == 1:
            tempFolderNameList = filename.split('_')
            dataFolderFullName = dataFolderName + '/' + tempFolderNameList[0] + tempFolderNameList[1] + tempFolderNameList[2]
            if not os.path.isdir(dataFolderFullName):
                os.makedirs(dataFolderFullName)
            #csv_file = filename #all fixed
            #zfile = zipfile.ZipFile(name)
            with gzip.open(name,'rb') as fin: #don't forget this line!
                for line in fin:
                    data=line.split('|')
                    dateframe=data[0].split()
                    timeframe=dateframe[1].split('.')
                    if len(timeframe) == 1:
                        timeframe = timeframe[0].split('+')
                    if latlongFilter(float(data[9]),float(data[10]),latN,latS,longE,longW):
                        writeFileName = dataFolderFullName + '/' + data[1] + '.txt'
                        testfile=open(writeFileName,'a')
                        testfile.write(dateframe[0]+','+timeframe[0]+','+data[1]+','+data[9]+','+data[10].split('\n')[0]+',' + data[8] + '\n')	
                        testfile.close		
    return

def readDFfromTxt(filename,pathname):
    '''
    Import text file as pandas.dataframe
    Parameters:
    -------------
    filename: input file name
    
    Returns:
    -------------
    data: pandas.dataframe including all the filtered GPS data points
    '''
    pointsfilename_full = pathname + '/' + filename
    data = pd.read_csv(pointsfilename_full,header = None, names = ['date','time', 'ID', 'lat','long', 'speed'])
    return data

def addEndPointLabel(df,timeDiffThres):
    '''
    Add label to the GPS data file indicating start and end point of trips
    Parameters:
    -------------
    df: GPS dataframe
    timeDiffThres: time difference threshold used for seperating trips

    Returns:
    -------------
    None. endPointLable is added to the input dataframe
    '''
    df['endPoint'] = None
    df.iloc[0,7] = 1
    for i in range(1,df.shape[0]):
        if i == df.shape[0] - 1:
            time1 = df.iloc[i-1,6]
            time2 = df.iloc[i,6]
            diff = time2 - time1
            if diff > timeDiffThres:
                if pd.isnull(df.iloc[i-1,7]):
                    df.iloc[i-1,7] = 2
                    df.iloc[i,7] = 9
                else:
                    df.iloc[i-1,7] = 9
                    df.iloc[i,7] = 9
            else:
                df.iloc[i,7] = 2
        else:
            time1 = df.iloc[i-1,6]
            time2 = df.iloc[i,6]
            diff = time2 - time1
            if diff > timeDiffThres:
                if pd.isnull(df.iloc[i-1,7]):
                    df.iloc[i-1,7] = 2
                    df.iloc[i,7] = 1
                else:
                    df.iloc[i-1,7] = 9
                    df.iloc[i,7] = 1
    return

def addDistanceToCenter(df,latC,longC):
    '''
    Add disToC column which calculates the distance from each point to the center point
    Parameters:
    -------------
    df: GPS dataframe
    latC,longC: lat long of the intersection center
    
    Returns:
    -------------
    None. The disToC column is added to the input df
    '''
    df['disToC'] = None
    for i in range(0,df.shape[0]):
        df.iloc[i,8] = haversine(df.iloc[i,3], df.iloc[i,4], latC,longC)
    return

def findDelayEndPointByDist(df):
    '''
    Find the first point passing the intersection
    Parameters:
    -------------
    df: GPS dataframe

    Returns:
    -------------
    tempFlag == : a boolean indicating whether this trip is valid or not, true is valid, false is not
    tempLat,tempLong: lat long of the first point passing the intersection
    
    Logic:
    The first point passing the intersection is found by distance to the intersection. If distance decrease first then increase, then the first point increasing is the point we need; 
    '''
    df['flag'] = None
    df.iloc[0,9] = 0
    for i in range(1,df.shape[0]):
        dist1 = df.iloc[i-1,8]
        dist2 = df.iloc[i,8]
        if abs(dist1-dist2) < 2:
            df.iloc[i,9] = 0
        elif dist1 > dist2:
            df.iloc[i,9] = 1
        else:
            df.iloc[i,9] = 2
    #index1 = np.where(df.flag == 2)[0][0]
    temp = np.where(df.flag == 2)[0]
    if len(temp) == 0:
        index1 = df.shape[0]-1
    else:
        index1 = np.where(df.flag == 2)[0][0]
    tempFlag1 = 0
    for i in range(0,index1):
        if df.iloc[i,9] == 1:
            tempFlag1 = 1
    if tempFlag1 == 0 or pd.isnull(index1):
        index1 = df.shape[0]-1
    tempLat = df.iloc[index1,3]
    tempLong = df.iloc[index1,4]
    tempTime = df.iloc[index1,1]
    tempFlag = 0
    for i in range(index1,df.shape[0]):
        if df.iloc[i,9] == 1:
            tempFlag = 1
    return [tempFlag==0,df.index.values[index1],tempLat,tempLong,tempTime]

def findProperRegion(lat1,long1,lat2,long2,lat3,long3):
    '''
    Expand the region a little bit
    Parameters:
    -------------
    lat1,long1,lat2,long2: lat long of the first point and second point

    Returns:
    -------------
    [latMin,longMin,latMax,longMax]: rectangle boundary
    '''
    latMin = min(lat1,lat2,lat3)
    latMax = max(lat1,lat2,lat3)
    longMin = min(long1,long2,long3)
    longMax = max(long1,long2,long3)
    latMin = latMin - 0.00005
    latMax = latMax + 0.00005
    longMin = longMin - 0.00005
    longMax = longMax + 0.00005
    #if haversine(latMin,0,latMax,0)<= 10:
    #    latMin = latMin - 0.00005
    #    latMax = latMax + 0.00005
    #if haversine(0,longMin,0,longMax) <= 10:
    #    longMin = longMin - 0.00005
    #    longMax = longMax + 0.00005
    return [latMin,longMin,latMax,longMax]
def findRectangleBufferBasedOnRegion(region,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC):
    '''
    Find the rectangle boudary for the specified region
    Parameters:
    -------------
    region: number representing different region
        west-1
        east-2
        north-3
        south-4
    latW,longW,latE,longE,latN,longN,latS,longS,latC,longC: lat long information

    Returns:
    -------------
    [latMin,longMin,latMax,longMax]: rectangle boundary
    '''
    if region == 1:
        tempLat = latW
        tempLong = longW
        tempCenLat = westcenlat
        tempCenLong = westcenlon
    elif region == 2:
        tempLat = latE
        tempLong = longE
        tempCenLat = eastcenlat
        tempCenLong = eastcenlon
    elif region == 3:
        tempLat = latN
        tempLong = longN
        tempCenLat = northcenlat
        tempCenLong = northcenlon
    else:
        tempLat = latS
        tempLong = longS
        tempCenLat = southcenlat
        tempCenLong = southcenlon
    #print tempCenLat, tempCenLong, tempLat,tempLong,latC,longC
    [latMin,longMin,latMax,longMax] = findProperRegion(tempCenLat, tempCenLong, tempLat,tempLong,latC,longC)
    return [latMin,longMin,latMax,longMax]
            
def findDelayEndPointByArea(df,startRegion,endRegion,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC) :
    '''
    Find the first point passing the intersection by rectangle buffer
    Parameters:
    -------------
    df: GPS dataframe
    stratRegion:number representing start region
        west-1
        east-2
        north-3
        south-4
    endRegion:number representing end region
    latW,longW,latE,longE,latN,longN,latS,longS,latC,longC: lat long information
    Returns:
    -------------
    tempFlag == : a boolean indicating whether this trip is valid or not, true is valid, false is not
    index1, tempLat,tempLong,tempTime: index number, lat, long, time of the first point passing the intersection
    index2: index number of the point before passing the intersection
    
    Logic:
    The first pointfindDelayEndPointByArea passing the intersection is found by rectangle area boundary for different region.
    '''
    [startlatMin,startlongMin,startlatMax,startlongMax] = findRectangleBufferBasedOnRegion(startRegion,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC)
    [endlatMin,endlongMin,endlatMax,endlongMax] = findRectangleBufferBasedOnRegion(endRegion,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC)
    #if startRegion == 4 and endRegion == 1:
    #    print startRegion,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC
    #    print startRegion, startlatMin,startlongMin,startlatMax,startlongMax
    for i in df.index.values:
        if latlongFilter(df.loc[i,'lat'],df.loc[i,'long'],startlatMin,startlatMax,startlongMin,startlongMax):
            index2 = i
            #if startRegion == 4 and endRegion == 1:
            #    print i, df.loc[i,'lat'],df.loc[i,'long'], 'zone:',startRegion
        else:
            if latlongFilter(df.loc[i,'lat'],df.loc[i,'long'],endlatMin,endlatMax,endlongMin,endlongMax):
                flag = endRegion
                index1 = i
                tempLat = df.loc[index1,'lat']
                tempLong = df.loc[index1,'long']
                tempTime = df.loc[index1,'time']
                #if startRegion == 4 and endRegion == 1:
                #    print i,df.loc[i,'lat'],df.loc[i,'long'], 'zone:',endRegion
                return [True,index1,tempLat,tempLong,tempTime,index2]
            else:
                #if startRegion == 4 and endRegion == 1:
                #    print i,df.loc[i,'lat'],df.loc[i,'long'], 'zone:','None'
                #print 'In loop:false'
                return [False,-1,-1,-1,-1,-1]
    print index2
    if 'index1' in locals():
        tempLat = df.loc[index1,'lat']
        tempLong = df.loc[index1,'long']
        tempTime = df.loc[index1,'time']
        return [True,index1,tempLat,tempLong,tempTime,index2]
    else:
        #print 'Out loop:false'
        return [False,-1,-1,-1,-1,-1]
            
        
def findDelayEndPointByArea_NotTooMuchChecking(df,startRegion,endRegion,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC) :
    '''
    Find the first point passing the intersection by rectangle buffer
    Parameters:
    -------------
    df: GPS dataframe
    stratRegion:number representing start region
        west-1
        east-2
        north-3
        south-4
    endRegion:number representing end region
    latW,longW,latE,longE,latN,longN,latS,longS,latC,longC: lat long information
    Returns:
    -------------
    tempFlag == : a boolean indicating whether this trip is valid or not, true is valid, false is not
    index1, tempLat,tempLong,tempTime: index number, lat, long, time of the first point passing the intersection
    index2: index number of the point before passing the intersection
    
    Logic:
    The first pointfindDelayEndPointByArea passing the intersection is found by rectangle area boundary for different region.
    '''
    [startlatMin,startlongMin,startlatMax,startlongMax] = findRectangleBufferBasedOnRegion(startRegion,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC)
    [endlatMin,endlongMin,endlatMax,endlongMax] = findRectangleBufferBasedOnRegion(endRegion,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC)
    flag = startRegion
    for i in df.index.values:
        if flag == startRegion:
            if latlongFilter(df.loc[i,'lat'],df.loc[i,'long'],startlatMin,startlatMax,startlongMin,startlongMax):
                df.loc[i,'regionFlag'] = flag
                index2 = i
                index1 = i
            else:
                index1 = i
                flag = endRegion
    if 'index1' in locals():
        tempLat = df.loc[index1,'lat']
        tempLong = df.loc[index1,'long']
        tempTime = df.loc[index1,'time']
        return [True,index1,tempLat,tempLong,tempTime,index2]
    else:
        return [False,-1,-1,-1,-1,-1]

def addSpeed(df):
    '''
    Calculate speed based on distance and time elapsed
    Parameters:
    -------------
    df: GPS dataframe

    Returns:
    -------------
    None
    '''
    for i in range(1,df.shape[0]):
        timeDiff = try_parsing_time(df.iloc[i,1]) - try_parsing_time(df.iloc[i-1,1])
        dis = haversine(df.iloc[i,3], df.iloc[i,4], df.iloc[i-1,3], df.iloc[i-1,4])
        df.iloc[i,5] = dis/(timeDiff.seconds+ 0.000001)
    return

def segmentLength(region, latW,longW,latE,longE,latN,longN,latS,longS,latC,longC):
    '''
    Calculate the segment length for specified region
    Parameters:
    -------------
    region: number representing different region
        west-1
        east-2
        north-3
        south-4
    latW,longW,latE,longE,latN,longN,latS,longS,latC,longC: lat long information
    Returns:
    -------------
    float: segment length
    '''
    if region == 1:
        return haversine(latW,longW, latC,longC)
    elif region == 2:
        return haversine(latE,longE, latC,longC)
    elif region == 3:
        return haversine(latN,longN, latC,longC)
    else:
        return haversine(latS,longS, latC,longC)

def defineIntersectionRegion(latC,longC,buffer = 0.0002):
    return [latC-buffer, latC+buffer, longC-buffer, longC+buffer]

def findIntersectionAndQueuePoint(df, startRegion, latW,longW,latE,longE,latN,longN,latS,longS,latC,longC):
    '''
    Find the queue point
    Parameters:
    -------------
    df: GPS dataframe
    startRegion: number representing the start region
        west-1
        east-2
        north-3
        south-4
    latW,longW,latE,longE,latN,longN,latS,longS,latC,longC: lat long information
    Returns:
    -------------
    [df.index.values[index],tempLat,tempLong,tempTime,tempSpeed,tempLength,tempPercentLength]: attributes of the first queue point including the index number, \
    lat, long, time, speed, distance to the intersection center, percent length compared to the segment length
    Notice 1: queue point is determined by speed, points that have speed less than 0.2m/s is considerred the queue point
    Notice 2: if there is not queue point, returns [-1,-1,-1,-1,-1,-1,-1]
    '''
    [lat1,lat2,long1,long2] = defineIntersectionRegion(latC,longC,0.0002)
    tempIntIndex = -1
    tempIntLat = -1
    tempIntLong = -1
    tempIntTime = -1
    tempIntSpeed = -1
    if pd.isnull(df.iloc[0,5]):
        addSpeed(df)
    for i in df.index.values:
        if latlongFilter(df.loc[i,'lat'],df.loc[i,'long'],lat1,lat2,long1,long2) and df.loc[i,'speed'] > 0:
            tempIntIndex = i
            tempIntLat = df.loc[i,'lat']
            tempIntLong = df.loc[i,'long']
            tempIntTime = df.loc[i,'time']
            tempIntSpeed = df.loc[i,'speed']
            break
    temp = np.where(df.speed < 0.2)[0]
    if len(temp) == 0:
        return [-1,-1,-1,-1,-1,-1,-1,tempIntIndex,tempIntLat,tempIntLong,tempIntTime,tempIntSpeed]
    index = temp[0]
    tempLat = df.iloc[index,3]
    tempLong = df.iloc[index,4]
    tempTime = df.iloc[index,1]
    tempSpeed = df.iloc[index,5]
    tempLength = haversine(tempLat, tempLong, latC,longC)
    segLength = segmentLength(startRegion, latW,longW,latE,longE,latN,longN,latS,longS,latC,longC)
    tempPercentLength = tempLength/segLength
    return [df.index.values[index],tempLat,tempLong,tempTime,tempSpeed,tempLength,tempPercentLength,tempIntIndex,tempIntLat,tempIntLong,tempIntTime,tempIntSpeed]
    
def generateDirectionNameList(n):
    tempList = []
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i!=j:
                tempName = str(i) + '_' + str(j)
                tempList.append(tempName)
    return tempList
    
def generateDirectionNameListFromLocation(w,e,n,s):
    tempList = []
    if w >0:
        tempList.append(1)
    if e> 0:
        tempList.append(2)
    if n> 0:
        tempList.append(3)
    if s> 0:
        tempList.append(4)
    tempListR = []
    for i in tempList:
        for j in tempList:
            if i!=j:
                tempName = str(i) + '_' + str(j)
                tempListR.append(tempName)
    return tempListR

def createTripsFile(tripsFolderName,dataFolderName,timeDiffThres, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed):
    '''
    Create Trips file
    Parameters:
    -------------
    inputfilename: GPS file name
    tripsFilename: Trips file name
    timeDiffThres: time difference threshold used for seperating trips
    westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon, latW,longW,latE,longE,latN,longN,latS,longS,latC,longC: lat long information
    Returns:
    -------------
    None.
    '''
    if not os.path.isdir(tripsFolderName):
        os.makedirs(tripsFolderName)
    tempList = generateDirectionNameListFromLocation(latW,latE,latN,latS)
    for i in tempList:
        tempPath = tripsFolderName + '/' + i
        if not os.path.isdir(tempPath):
            os.makedirs(tempPath)

    tempList = []
    for subdir, dirs, files in os.walk(dataFolderName):
        for file in files:
            tempList.append(os.path.join(subdir, file))
    
    num_cores = multiprocessing.cpu_count()
    ##if __name__ == '__main__':
    Parallel(n_jobs=num_cores)(delayed(addTripsToFile)(tripsFolderName,filename,timeDiffThres,westcenlat, westcenlon, eastcenlat,\
    eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,\
    latS,longS,latC,longC,EWspeed,NSspeed) for filename in tempList)  
    return
      
def createTripsFile_singleCore(tripsFolderName,dataFolderName,timeDiffThres, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed):
    '''
    Create Trips file
    Parameters:
    -------------
    inputfilename: GPS file name
    tripsFilename: Trips file name
    timeDiffThres: time difference threshold used for seperating trips
    westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon, latW,longW,latE,longE,latN,longN,latS,longS,latC,longC: lat long information
    Returns:
    -------------
    None.
    '''
    if not os.path.isdir(tripsFolderName):
        os.makedirs(tripsFolderName)
    tempList = generateDirectionNameListFromLocation(latW,latE,latN,latS)
    for i in tempList:
        tempPath = tripsFolderName + '/' + i
        if not os.path.isdir(tempPath):
            os.makedirs(tempPath)

    tempList = []
    for subdir, dirs, files in os.walk(dataFolderName):
        for file in files:
            tempList.append(os.path.join(subdir, file))
    
    for filename in tempList:
        addTripsToFile(tripsFolderName,filename,timeDiffThres,westcenlat, westcenlon, eastcenlat,\
        eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,\
        latS,longS,latC,longC,EWspeed,NSspeed)
    return

def createTripsFile_new(tripsFolderName,dataFolderName,timeDiffThres, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed):
    '''
    Create Trips file
    Parameters:
    -------------
    inputfilename: GPS file name
    tripsFilename: Trips file name
    timeDiffThres: time difference threshold used for seperating trips
    westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon, latW,longW,latE,longE,latN,longN,latS,longS,latC,longC: lat long information
    Returns:
    -------------
    None.
    '''
    if not os.path.isdir(tripsFolderName):
        os.makedirs(tripsFolderName)
    tempList = generateDirectionNameList(4)
    for i in tempList:
        tempPath = tripsFolderName + '/' + i
        if not os.path.isdir(tempPath):
            os.makedirs(tempPath)

    tempList = []
    for subdir, dirs, files in os.walk(dataFolderName):
        for file in files:
            tempList.append(os.path.join(subdir, file))
    num_cores = multiprocessing.cpu_count()
    #if __name__ == '__main__':
    Parallel(n_jobs=num_cores)(delayed(addTripsToFile)(tripsFolderName,filename,timeDiffThres,westcenlat, westcenlon, eastcenlat,\
    eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,\
    latS,longS,latC,longC,EWspeed,NSspeed) for filename in tempList)  
    return

def addTripsToFile(tripsFolderName,filename,timeDiffThres,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed):
    '''
    Add trip to Trips txt file
    Parameters:
    -------------
    ID: vehicle ID
    df: GPS dataframe
    TripsDF: trips dataframe
    indexNum: current row number waiting for inputs
    timeDiffThres: time difference threshold used for seperating trips
    westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS: lat long information
    Returns:
    -------------
    None
    '''
    df = pd.read_csv(filename,header = None, names = ['date','time', 'ID', 'lat','long',  'speed'])
    if df.shape[0]<3:
        return
    #Create timeDate to represent the time of day, and sort the data by time of day
    df['timeObj'] = None
    for i in df.index.values:
        df.loc[i,'timeObj'] = try_parsing_time(df.loc[i,'time'])
    #Depends on the data, decides whether or not sort the data
    df = df.sort(['timeObj'],ascending=[1])    
    addEndPointLabel(df,timeDiffThres)
    k = 0
    #debugFileName = tripsFolderName + '/' + 'debug.txt'
    #debug=open(debugFileName,'a')
    
    while (k<df.shape[0]):
        p = k
        while p < df.shape[0] and (df.iloc[p,7] != 1):
            p+=1
        n1 = p
        while p < df.shape[0] and (df.iloc[p,7] != 2):
            p+=1
        n2 = p
        if n2>n1: 
            tempIndex1= df.index.values[n1]
            outID = df.loc[tempIndex1,'ID']
            outStartIndex = tempIndex1
            outStartDate = df.loc[tempIndex1,'date']
            outStartTime = df.loc[tempIndex1,'time']
            outStartLat = df.loc[tempIndex1,'lat']
            outStartLong = df.loc[tempIndex1,'long']
            outStartSpeed = df.loc[tempIndex1,'speed']
            tempIndex2= df.index.values[n2]
            outEndIndex = tempIndex2
            outEndDate = df.loc[tempIndex2,'date']
            outEndTime = df.loc[tempIndex2,'time']
            outEndLat = df.loc[tempIndex2,'lat']
            outEndLong = df.loc[tempIndex2,'long']
            outEndSpeed = df.loc[tempIndex2,'speed']
            outStartRegion = findRegion(outStartLat,outStartLong,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS )
            outEndRegion = findRegion(outEndLat,outEndLong,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS )
            #if outStartRegion == 2 and outEndRegion == 4:
            #    debug.write('Start:'+str(outStartRegion)+'    End:'+str(outEndRegion) + '    File:' + filename + '\n')
            #    print 'Start:',outStartRegion,'    End:',outEndRegion
            if (not pd.isnull(outStartRegion)) and (not pd.isnull(outEndRegion)):
                if outStartRegion!=outEndRegion:
                    [tempEffective,tempDelayIndex,tempLat,tempLong,tempTime,tempRegionEndIndex] = findDelayEndPointByArea(df.iloc[n1:n2+1],outStartRegion,outEndRegion,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC)
                    #if outStartRegion == 2 and outEndRegion == 4:
                    #    debug.write('Effective:'+str(tempEffective) + '\n')
                    #    print df.iloc[n1:n2+1]
                    if tempEffective:
                        outDelayCalIndex = tempDelayIndex
                        outDelayCalLat = tempLat
                        outDelayCalLong = tempLong
                        outDelayCalTime = tempTime
                        outDelay = delayCalculation(outStartTime, outStartLat,outStartLong,outStartRegion, outDelayCalTime, outDelayCalLat,outDelayCalLong,outEndRegion,latC,longC,EWspeed,NSspeed)
                        #debug.write('Delay:'+str(outDelay) + '\n')
                        if outDelay < 600:
                            n3 = np.where(df.index.values == tempRegionEndIndex)[0][0]
                            [tempQueueIndex,tempQueueLat,tempQueueLong,tempQueueTime,tempQueueSpeed,tempLength,tempPercentLength,tempIntIndex,tempIntLat,tempIntLong,tempIntTime,tempIntSpeed] = findIntersectionAndQueuePoint(df.iloc[n1:n3+1], outStartRegion, latW,longW,latE,longE,latN,longN,latS,longS,latC,longC)
                            
                            minGroup = 10
                            tempTime = outStartTime
                            tempMinInt = int(tempTime.split(':')[1])
                            tempHourInt = int(tempTime.split(':')[0])
                            tempGroup = tempHourInt* 60/ minGroup + tempMinInt/ minGroup
                            
                            writeFileName = tripsFolderName + '/' + str(outStartRegion) + '_' + str(outEndRegion) + '/' + str(tempGroup) + '.txt'
                            outfile=open(writeFileName,'a')
                            outfile.write(str(outID) + ','\
                            + str(outStartIndex) + ','\
                            + outStartDate + ','\
                            + outStartTime + ','\
                            + str(outStartLat) + ','\
                            + str(outStartLong) + ','\
                            + str(outStartSpeed) + ','\
                            + str(outEndIndex) + ','\
                            + outEndDate + ','\
                            + outEndTime + ','\
                            + str(outEndLat) + ','\
                            + str(outEndLong) + ','\
                            + str(outEndSpeed) + ','\
                            + str(outStartRegion) + ','\
                            + str(outEndRegion) + ','\
                            + str(outDelayCalIndex) + ','\
                            + str(outDelayCalLat) + ','\
                            + str(outDelayCalLong) + ','\
                            + str(outDelayCalTime) + ','\
                            + str(outDelay) + ','\
                            + str(tempQueueIndex) + ','\
                            + str(tempQueueLat) + ','\
                            + str(tempQueueLong) + ','\
                            + str(tempQueueTime) + ','\
                            + str(tempQueueSpeed) + ','\
                            + str(tempLength) + ','\
                            + str(tempPercentLength) + ','\
                            + str(tempIntIndex) + ','\
                            + str(tempIntLat) + ','\
                            + str(tempIntLong) + ','\
                            + str(tempIntTime) + ','\
                            + str(tempIntSpeed) + '\n')
                            outfile.close
        #debug.close
        k = p+1
    return
    
def findRegion(lat,lon,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS ):
    '''
    Return numbers for different regions
    Parameters:
    -------------
    lat,lon,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS: lat long information

    Returns:
    -------------
    int: number representing different region
        west-1
        east-2
        north-3
        south-4
    '''
    if latlongFilter(lat,lon,westcenlat,latW,westcenlon,longW):
        return 1
    elif latlongFilter(lat,lon,eastcenlat,latE,eastcenlon,longE):
        return 2
    elif latlongFilter(lat,lon,northcenlat,latN,northcenlon,longN):
        return 3
    elif latlongFilter(lat,lon,southcenlat,latS,southcenlon,longS):
        return 4
def addRegionTags(TripsDF, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
latW,longW,latE,longE,latN,longN,latS,longS):
    '''
    Add region tags and effectiveTrip label to the TripsDF
    Parameters:
    -------------
    TripsDF: trips dataframe
    westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS: lat long information

    Returns:
    -------------
    TripsDF_Tagged: TripsDF with region tags and effectiveTrip label
    '''
    TripsDF['startRegion'] = None
    TripsDF['endRegion'] = None
    TripsDF['effectiveTrip'] = None
    for i in TripsDF.index.values:
        TripsDF.loc[i,'startRegion'] = findRegion(TripsDF.loc[i,'startLat'],TripsDF.loc[i,'startLong'],westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS )
        TripsDF.loc[i,'endRegion'] = findRegion(TripsDF.loc[i,'endLat'],TripsDF.loc[i,'endLong'],westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS )
        if (not pd.isnull(TripsDF.loc[i,'startRegion'])) and (not pd.isnull(TripsDF.loc[i,'endRegion'])):
            if TripsDF.loc[i,'startRegion']!=TripsDF.loc[i,'endRegion']:
                TripsDF.loc[i,'effectiveTrip'] =1
    return TripsDF
def keepEffectiveTrips(TripsDF_Tagged):
    '''
    Only keep effective trips
    Parameters:
    -------------
    TripsDF_Tagged: TripsDF with region tags and effectiveTrip label

    Returns:
    -------------
    TripsDF_Tagged: TripsDF with only effective trips
    '''
    TripsDF_Tagged = TripsDF_Tagged[pd.isnull(TripsDF_Tagged.effectiveTrip) == False]
    TripsDF_Tagged = TripsDF_Tagged.reset_index(drop=True)
    del TripsDF_Tagged['effectiveTrip']
    return TripsDF_Tagged
    
def timeCalculate(lat,lon,region,latC,longC,EWspeed,NSspeed):
    '''
    Calculate freeflow travel time from a point to intersection point
    Parameters:
    -------------
    lat, lon: lat long information of the point
    region: region number
    latC, longC: lat long information of the intersection center
    EWspeed,NSspeed: speed on roadways

    Returns:
    -------------
    dis/1000/tempSpeed*60*60: time in second
    '''
    if region == 1 or region == 2:
        tempSpeed = EWspeed
    else:
        tempSpeed = NSspeed
    dis = haversine(lat, lon, latC,longC)
    return dis/1000/tempSpeed*60*60

def delayCalculation(startTime, startLat,startLong,startRegion, delayCalTime, delayCalLat,delayCalLong,endRegion ,latC,longC,EWspeed,NSspeed):
    '''
    Calculate delay for each trip
    Parameters:
    -------------
    TripsDF: trips dataframe
    latC,longC: lat and long for intersection centroid
    EWspeed,NSspeed: speed for EW raod and NS road in KM/h

    Returns:
    -------------
    TripsDF: trips dataframe with delay information
    '''
    tempTime1 = timeCalculate(startLat,startLong,startRegion,latC,longC,EWspeed,NSspeed)
    tempTime2 = timeCalculate(delayCalLat,delayCalLong,endRegion,latC,longC,EWspeed,NSspeed)
    #tempTime1 = timeCalculate(TripsDF.loc[i,'startLat'],TripsDF.loc[i,'startLong'],TripsDF.loc[i,'startRegion'],latC,longC,EWspeed,NSspeed)
    #tempTime2 = timeCalculate(TripsDF.loc[i,'endLat'],TripsDF.loc[i,'endLong'],TripsDF.loc[i,'endRegion'],latC,longC,EWspeed,NSspeed)
    tempFreeTime = tempTime1 + tempTime2
    tempActualTimeStart = try_parsing_time(startTime)
    tempActualTimeEnd = try_parsing_time(delayCalTime)
    tempActualTime = (tempActualTimeEnd - tempActualTimeStart).total_seconds()
    tempDelay = tempActualTime - tempFreeTime
    if tempDelay < 0:
        tempDelay = 0
    return tempDelay

def addDelay(TripsDF,latC,longC,EWspeed,NSspeed):
    '''
    Calculate delay for each trip
    Parameters:
    -------------
    TripsDF: trips dataframe
    latC,longC: lat and long for intersection centroid
    EWspeed,NSspeed: speed for EW raod and NS road in KM/h

    Returns:
    -------------
    TripsDF: trips dataframe with delay information
    '''
    print 'Add information to trips file: add delay'
    TripsDF['delay'] = None
    counter = 0
    for i in TripsDF.index.values:
        counter += 1
        if counter%1000 == 0:
            print 'Add information to trips file: add delay',counter,'/',TripsDF.shape[0]
        tempTime1 = timeCalculate(TripsDF.loc[i,'startLat'],TripsDF.loc[i,'startLong'],TripsDF.loc[i,'startRegion'],latC,longC,EWspeed,NSspeed)
        tempTime2 = timeCalculate(TripsDF.loc[i,'delayCalLat'],TripsDF.loc[i,'delayCalLong'],TripsDF.loc[i,'endRegion'],latC,longC,EWspeed,NSspeed)
        #tempTime1 = timeCalculate(TripsDF.loc[i,'startLat'],TripsDF.loc[i,'startLong'],TripsDF.loc[i,'startRegion'],latC,longC,EWspeed,NSspeed)
        #tempTime2 = timeCalculate(TripsDF.loc[i,'endLat'],TripsDF.loc[i,'endLong'],TripsDF.loc[i,'endRegion'],latC,longC,EWspeed,NSspeed)
        tempFreeTime = tempTime1 + tempTime2
        tempActualTimeStart = try_parsing_time(TripsDF.loc[i,'startTime'])
        tempActualTimeEnd = try_parsing_time(TripsDF.loc[i,'delayCalTime'])
        tempActualTime = (tempActualTimeEnd - tempActualTimeStart).total_seconds()
        tempDelay = tempActualTime - tempFreeTime
        if tempDelay < 0:
            tempDelay = 0
        TripsDF.loc[i,'delay'] = tempDelay
    return TripsDF

def findTripFromGPSFile(df,TripsDF,tripIndex):
    '''
    Find corresponding GPS records from GPS file
    Parameters:
    -------------
    df: GPS dataframe
    TripsDF: trips dataframe
    tripIndex: index in TripsDF

    Returns:
    -------------
    pandas.dataframe: GPS dataframe corresponding to the trip
    Notice that a column named flag is added to the DF where 0- normal nodes, 1- trip start node,
    2- trip end node, 3- node where vehicle just passes the intersection, 
    9- indicating both 2 and 3 (the trip end node is the node where vehicle just passes the intersection)
    '''
    temp = tripIndex
    startIndexTemp = TripsDF.loc[temp,'startIndex']
    endIndexTemp = TripsDF.loc[temp,'endIndex']
    delayCalIndexTemp = TripsDF.loc[temp,'delayCalIndex']
    dateTemp = TripsDF.loc[temp,'startDate']
    IDTemp = TripsDF.loc[temp,'ID']
    gb = df.groupby(['ID'])
    dic = dict(list(gb))
    dfI = dic[IDTemp]
    gbSameday = dfI.groupby(['date'])
    groupsSameday = dict(list(gbSameday))
    dfISameday = groupsSameday[dateTemp]
    dfISameday['timeObj'] = None
    for i in dfISameday.index.values:
        dfISameday.loc[i,'timeObj'] = try_parsing_time(dfISameday.loc[i,'time'])
    dfISameday = dfISameday.sort(['timeObj'],ascending=[1])
    index1 = np.where(dfISameday.index.values == startIndexTemp)[0][0]
    index2 = np.where(dfISameday.index.values == endIndexTemp)[0][0]
    index3 = np.where(dfISameday.index.values == delayCalIndexTemp)[0][0]
    dfISameday['flag'] = 0
    dfISameday.iloc[index1,7] = 1
    if index2 == index3:
        dfISameday.iloc[index2,7] = 9
    else:
        dfISameday.iloc[index2,7] = 2
        dfISameday.iloc[index3,7] = 3
    return dfISameday.iloc[index1:(index2+1),]

def createTripGPSDF(TripGPSDFfilename, pathname, df,TripsDF):
    '''
    This function is able to create the GPS file based on trip sequence
    Parameters:
    -------------
    TripGPSDFfilename: TripGPSDF file name
    pathname: path name
    df: GPS dataframe
    TripsDF: trips dataframe

    Returns:
    -------------
    None. TripGPSDF is saved as csv file
    '''
    fullname = pathname + '/' + TripGPSDFfilename
    totalNum = len(TripsDF.index.values)
    counter = 0
    for i in TripsDF.index.values:
        counter += 1
        print totalNum,':',counter
        if 'TripGPSDF' in locals():
            TripGPSDF = TripGPSDF.append(findTripFromGPSFile(df,TripsDF,i))
        else:
            TripGPSDF = findTripFromGPSFile(df,TripsDF,i)
    TripGPSDF.to_csv(fullname)
    return

def plotStats(folderName,pathname,latW,latE,latN,latS,minGroup = 10):
    '''
    This function is to calculate the mean value of delay and queuePercentageLength for different time of day and different directions. It also generates plots
    Parameters:
    -------------
    TripsDF: trips dataframe
    filenameString: the name string you want to include in the file names
    minGroup: time interval you want to use to aggregate the data. Default is 10min, which means that 1 hour will be segmented as 6 segments    

    Returns:
    -------------
    None.
    
    Outputs:
    -------------
    1 file named _stats.csv: this is the file have the mean values () of all time periods for all directions. Notice that the timeGroup variable indicates time of day. \
    If minGroup = 10, then it is calculated by hour * 6 +  min/10. So it's ranging from 0 to 143.
    A group of csv files named as (a, b)_*Time.csv: a is the startRegion, b is the endRegion. This file has information for the specified direction
        west-1
        east-2
        north-3
        south-4
    A group of pdf files named as Delay(a, b)_*Time.pdf: delay plot for the specified direction
    A group of pdf files named as Queue(a, b)_*Time.pdf: Queue plot for the specified direction. Notice the queue is the percentage of queue length to the segment length
    '''
    folderFullName = pathname + '/' + folderName
    tripsFolderName = folderFullName + '/trips'
    plotFolderName = folderFullName + '/plots'
    tempList = generateDirectionNameListFromLocation(latW,latE,latN,latS)
    if not os.path.isdir(plotFolderName):
        os.makedirs(plotFolderName)
    for nameIte in tempList:
        print 'Create plot for ', nameIte
        tripsFolderName1 = tripsFolderName + '/' + nameIte
        plotFolderName1 = plotFolderName + '/' + nameIte
        timeGroupList = []
        timeList = []
        delayMeanList = []
        delayStdList = []
        delay75List = []
        delay25List = []
        queue95List = []
        queueMaxList = []
        sampleList = []
        if len(os.listdir(tripsFolderName1)) > 0:
            for i in os.listdir(tripsFolderName1):
                timeGroupList.append(int(i.split('.')[0]))
                filename = tripsFolderName1 + '/' + i
                TripsDF = pd.read_csv(filename,header = None, names = ['ID','startIndex','startDate','startTime','startLat','startLong','startSpeed','endIndex','endDate','endTime','endLat','endLong','endSpeed','startRegion','endRegion','delayCalIndex','delayCalLat','delayCalLong','delayCalTime','delay','queueIndex','queueLat','queueLong','queueTime','queueSpeed','queueLength','queuePercentageLength','intersectionIndex','intersectionLat','intersectionLong','intersectionTime','intersectionSpeed'])
                delayMeanList.append(TripsDF['delay'].mean())
                
                delayStdList.append(TripsDF['delay'].std())
                delay75List.append(TripsDF['delay'].quantile(0.75))
                delay25List.append(TripsDF['delay'].quantile(0.25))
                temp = TripsDF['queuePercentageLength'].quantile(0.95)*100
                if temp < 0:
                    temp = 0
                queue95List.append(temp)
                temp = TripsDF['queuePercentageLength'].max()*100
                if temp < 0:
                    temp = 0
                queueMaxList.append(temp)
                temp = int(i.split('.')[0])
                minGroup = 10
                timeList.append(time(temp/(60/minGroup),temp%(60/minGroup)*minGroup))
                sampleList.append(TripsDF.shape[0])
            n1 = int(nameIte[0])
            n2 = int(nameIte[2])
            if n1 == 1:
                namePart1 = 'Eastbound'
                if n2 == 2:
                    namePart2 = 'through'
                elif n2 == 3:
                    namePart2 = 'left turn'
                else:
                    namePart2 = 'right turn'
            elif n1 == 2:
                namePart1 = 'Westbound'
                if n2 == 1:
                    namePart2 = 'through'
                elif n2 == 3:
                    namePart2 = 'right turn'
                else:
                    namePart2 = 'left turn'
            elif n1 == 3:
                namePart1 = 'Southbound'
                if n2 == 1:
                    namePart2 = 'right turn'
                elif n2 == 2:
                    namePart2 = 'left turn'
                else:
                    namePart2 = 'through'
            else:
                namePart1 = 'Northbound'
                if n2 == 1:
                    namePart2 = 'left turn'
                elif n2 == 2:
                    namePart2 = 'right turn'
                else:
                    namePart2 = 'through'
    
            xtickList = []
            for i in range(0,24,2):
                xtickList.append(time(i))   
            
            name2 = plotFolderName + '/' + folderName + '_Delay' + nameIte + '.pdf'
            ts = pd.Series(delayMeanList,index = timeList)
            #ts1 = pd.Series(delay75List,index = timeList)
            #ts2 = pd.Series(delay25List,index = timeList)
            tsErr = pd.Series(delayStdList,index = timeList)
            plt.figure()
            plt.ylabel('Delay (s)')
            plt.xlabel('Time')
            plt.title(namePart1 + ' ' + namePart2 + ' delay')
            ts.plot(rot=30,yerr = tsErr,xticks = xtickList,fmt='o', ecolor='g')
            #ts.plot(rot=30)
            #ts1.plot(rot=30)
            #ts2.plot(rot=30,xticks = xtickList)
            #plt.legend()
            plt.savefig(name2)
            plt.clf()
    
            name3 = plotFolderName + '/' + folderName + '_Queue' + nameIte + '.pdf'
            ts = pd.Series(queue95List,index = timeList)
            ts1 = pd.Series(queueMaxList,index = timeList)
            plt.figure()
            plt.ylabel('Queue length percentage (%)')
            plt.xlabel('Time')
            plt.title(namePart1 + ' ' + namePart2 + ' queue')
            ts.plot(rot=30, label = '95 Percentile')
            ts1.plot(rot=30, label = 'Max',xticks = xtickList)
            plt.legend()
            plt.savefig(name3)
            plt.clf()
            
            name4 = plotFolderName + '/' + folderName + '_SampleSize' + nameIte + '.pdf'
            ts = pd.Series(sampleList,index = timeList)
            plt.figure()
            plt.ylabel('Sample size')
            plt.xlabel('Time')
            plt.title(namePart1 + ' ' + namePart2 + ' sample size')
            ts.plot(rot=30,xticks = xtickList)
            plt.savefig(name4)
            plt.clf()
    return

def plotStatsInOneFolder_eachSingleDay(plotFolderName, folderName,pathname,latW,latE,latN,latS,minGroup = 10):
    '''
    This function is to calculate the mean value of delay and queuePercentageLength for different time of day and different directions. It also generates plots
    Parameters:
    -------------
    TripsDF: trips dataframe
    filenameString: the name string you want to include in the file names
    minGroup: time interval you want to use to aggregate the data. Default is 10min, which means that 1 hour will be segmented as 6 segments    

    Returns:
    -------------
    None.
    
    Outputs:
    -------------
    1 file named _stats.csv: this is the file have the mean values () of all time periods for all directions. Notice that the timeGroup variable indicates time of day. \
    If minGroup = 10, then it is calculated by hour * 6 +  min/10. So it's ranging from 0 to 143.
    A group of csv files named as (a, b)_*Time.csv: a is the startRegion, b is the endRegion. This file has information for the specified direction
        west-1
        east-2
        north-3
        south-4
    A group of pdf files named as Delay(a, b)_*Time.pdf: delay plot for the specified direction
    A group of pdf files named as Queue(a, b)_*Time.pdf: Queue plot for the specified direction. Notice the queue is the percentage of queue length to the segment length
    '''
    folderFullName = pathname + '/' + folderName
    tripsFolderName = folderFullName + '/trips'
    if not os.path.isdir(plotFolderName):
        os.makedirs(plotFolderName)
    for dirname in os.listdir(tripsFolderName):
        tripsFolderName1 = tripsFolderName + '/' + dirname
        tempList = generateDirectionNameListFromLocation(latW,latE,latN,latS)
        for nameIte in tempList:
            print 'Create plot for ', dirname, nameIte
            tripsFolderName2 = tripsFolderName1 + '/' + nameIte
            timeGroupList = []
            timeList = []
            delayMeanList = []
            delayStdList = []
            delay75List = []
            delay25List = []
            queue95List = []
            queueMaxList = []
            sampleList = []
            if len(os.listdir(tripsFolderName2)) > 0:
                for i in os.listdir(tripsFolderName2):
                    timeGroupList.append(int(i.split('.')[0]))
                    filename = tripsFolderName2 + '/' + i
                    TripsDF = pd.read_csv(filename,header = None, names = ['ID','startIndex','startDate','startTime','startLat','startLong','startSpeed','endIndex','endDate','endTime','endLat','endLong','endSpeed','startRegion','endRegion','delayCalIndex','delayCalLat','delayCalLong','delayCalTime','delay','queueIndex','queueLat','queueLong','queueTime','queueSpeed','queueLength','queuePercentageLength','intersectionIndex','intersectionLat','intersectionLong','intersectionTime','intersectionSpeed'])
                    delayMeanList.append(TripsDF['delay'].mean())
                    
                    delayStdList.append(TripsDF['delay'].std())
                    delay75List.append(TripsDF['delay'].quantile(0.75))
                    delay25List.append(TripsDF['delay'].quantile(0.25))
                    temp = TripsDF['queuePercentageLength'].quantile(0.95)*100
                    if temp < 0:
                        temp = 0
                    queue95List.append(temp)
                    temp = TripsDF['queuePercentageLength'].max()*100
                    if temp < 0:
                        temp = 0
                    queueMaxList.append(temp)
                    temp = int(i.split('.')[0])
                    minGroup = 10
                    timeList.append(time(temp/(60/minGroup),temp%(60/minGroup)*minGroup))
                    sampleList.append(TripsDF.shape[0])
                n1 = int(nameIte[0])
                n2 = int(nameIte[2])
                if n1 == 1:
                    namePart1 = 'Eastbound'
                    if n2 == 2:
                        namePart2 = 'through'
                    elif n2 == 3:
                        namePart2 = 'left turn'
                    else:
                        namePart2 = 'right turn'
                elif n1 == 2:
                    namePart1 = 'Westbound'
                    if n2 == 1:
                        namePart2 = 'through'
                    elif n2 == 3:
                        namePart2 = 'right turn'
                    else:
                        namePart2 = 'left turn'
                elif n1 == 3:
                    namePart1 = 'Southbound'
                    if n2 == 1:
                        namePart2 = 'right turn'
                    elif n2 == 2:
                        namePart2 = 'left turn'
                    else:
                        namePart2 = 'through'
                else:
                    namePart1 = 'Northbound'
                    if n2 == 1:
                        namePart2 = 'left turn'
                    elif n2 == 2:
                        namePart2 = 'right turn'
                    else:
                        namePart2 = 'through'
        
                xtickList = []
                for i in range(0,24,2):
                    xtickList.append(time(i))   
                
                name2 = plotFolderName + '/' + folderName + dirname + '_Delay' + nameIte + '.pdf'
                ts = pd.Series(delayMeanList,index = timeList)
                #ts1 = pd.Series(delay75List,index = timeList)
                #ts2 = pd.Series(delay25List,index = timeList)
                tsErr = pd.Series(delayStdList,index = timeList)
                plt.figure()
                plt.ylabel('Delay (s)')
                plt.xlabel('Time')
                plt.title(namePart1 + ' ' + namePart2 + ' delay')
                ts.plot(rot=30,yerr = tsErr,xticks = xtickList,fmt='o', ecolor='g')
                #ts.plot(rot=30)
                #ts1.plot(rot=30)
                #ts2.plot(rot=30,xticks = xtickList)
                #plt.legend()
                plt.savefig(name2)
                plt.clf()
        
                name3 = plotFolderName + '/' + folderName + dirname + '_Queue' + nameIte + '.pdf'
                ts = pd.Series(queue95List,index = timeList)
                ts1 = pd.Series(queueMaxList,index = timeList)
                plt.figure()
                plt.ylabel('Queue length percentage (%)')
                plt.xlabel('Time')
                plt.title(namePart1 + ' ' + namePart2 + ' queue')
                ts.plot(rot=30, label = '95 Percentile')
                ts1.plot(rot=30, label = 'Max',xticks = xtickList)
                plt.legend()
                plt.savefig(name3)
                plt.clf()
                
                name4 = plotFolderName + '/' + folderName + dirname + '_SampleSize' + nameIte + '.pdf'
                ts = pd.Series(sampleList,index = timeList)
                plt.figure()
                plt.ylabel('Sample size')
                plt.xlabel('Time')
                plt.title(namePart1 + ' ' + namePart2 + ' sample size')
                ts.plot(rot=30,xticks = xtickList)
                plt.savefig(name4)
                plt.clf()
    return

def plotStatsInOneFolder_mergeMultipleDays(plotFolderName, folderName,pathname,latW,latE,latN,latS,minGroup = 10):
    '''
    This function is to calculate the mean value of delay and queuePercentageLength for different time of day and different directions. It also generates plots
    Parameters:
    -------------
    TripsDF: trips dataframe
    filenameString: the name string you want to include in the file names
    minGroup: time interval you want to use to aggregate the data. Default is 10min, which means that 1 hour will be segmented as 6 segments    

    Returns:
    -------------
    None.
    
    Outputs:
    -------------
    1 file named _stats.csv: this is the file have the mean values () of all time periods for all directions. Notice that the timeGroup variable indicates time of day. \
    If minGroup = 10, then it is calculated by hour * 6 +  min/10. So it's ranging from 0 to 143.
    A group of csv files named as (a, b)_*Time.csv: a is the startRegion, b is the endRegion. This file has information for the specified direction
        west-1
        east-2
        north-3
        south-4
    A group of pdf files named as Delay(a, b)_*Time.pdf: delay plot for the specified direction
    A group of pdf files named as Queue(a, b)_*Time.pdf: Queue plot for the specified direction. Notice the queue is the percentage of queue length to the segment length
    '''
    folderFullName = pathname + '/' + folderName
    tripsFolderName = folderFullName + '/trips'
    tempList = generateDirectionNameListFromLocation(latW,latE,latN,latS)
    if not os.path.isdir(plotFolderName):
        os.makedirs(plotFolderName)
    for nameIte in tempList:
        print 'Create plot for ', nameIte
        
        plotFolderName1 = plotFolderName + '/' + nameIte
        timeGroupList = []
        timeList = []
        delayMeanList = []
        delayStdList = []
        delay75List = []
        delay25List = []
        queue95List = []
        queueMaxList = []
        sampleList = []
        for dateName in os.listdir(tripsFolderName):
            tripsFolderName1 = tripsFolderName + '/' + dateName + '/' + nameIte
            if len(os.listdir(tripsFolderName1)) > 0:
                for i in os.listdir(tripsFolderName1):
                    timeGroupList.append(int(i.split('.')[0]))
                    filename = tripsFolderName1 + '/' + i
                    TripsDF = pd.read_csv(filename,header = None, names = ['ID','startIndex','startDate','startTime','startLat','startLong','startSpeed','endIndex','endDate','endTime','endLat','endLong','endSpeed','startRegion','endRegion','delayCalIndex','delayCalLat','delayCalLong','delayCalTime','delay','queueIndex','queueLat','queueLong','queueTime','queueSpeed','queueLength','queuePercentageLength','intersectionIndex','intersectionLat','intersectionLong','intersectionTime','intersectionSpeed'])
                    delayMeanList.append(TripsDF['delay'].mean())
                    
                    delayStdList.append(TripsDF['delay'].std())
                    delay75List.append(TripsDF['delay'].quantile(0.75))
                    delay25List.append(TripsDF['delay'].quantile(0.25))
                    temp = TripsDF['queuePercentageLength'].quantile(0.95)*100
                    if temp < 0:
                        temp = 0
                    queue95List.append(temp)
                    temp = TripsDF['queuePercentageLength'].max()*100
                    if temp < 0:
                        temp = 0
                    queueMaxList.append(temp)
                    temp = int(i.split('.')[0])
                    minGroup = 10
                    timeList.append(time(temp/(60/minGroup),temp%(60/minGroup)*minGroup))
                    sampleList.append(TripsDF.shape[0])
            n1 = int(nameIte[0])
            n2 = int(nameIte[2])
            if n1 == 1:
                namePart1 = 'Eastbound'
                if n2 == 2:
                    namePart2 = 'through'
                elif n2 == 3:
                    namePart2 = 'left turn'
                else:
                    namePart2 = 'right turn'
            elif n1 == 2:
                namePart1 = 'Westbound'
                if n2 == 1:
                    namePart2 = 'through'
                elif n2 == 3:
                    namePart2 = 'right turn'
                else:
                    namePart2 = 'left turn'
            elif n1 == 3:
                namePart1 = 'Southbound'
                if n2 == 1:
                    namePart2 = 'right turn'
                elif n2 == 2:
                    namePart2 = 'left turn'
                else:
                    namePart2 = 'through'
            else:
                namePart1 = 'Northbound'
                if n2 == 1:
                    namePart2 = 'left turn'
                elif n2 == 2:
                    namePart2 = 'right turn'
                else:
                    namePart2 = 'through'
    
            xtickList = []
            for i in range(0,24,2):
                xtickList.append(time(i))   

            if len(delayMeanList)>0:
                name2 = plotFolderName + '/' + folderName + '_Delay' + nameIte + '.pdf'
                ts = pd.Series(delayMeanList,index = timeList)
                #ts1 = pd.Series(delay75List,index = timeList)
                #ts2 = pd.Series(delay25List,index = timeList)
                tsErr = pd.Series(delayStdList,index = timeList)
                plt.figure()
                plt.ylabel('Delay (s)')
                plt.xlabel('Time')
                plt.title(namePart1 + ' ' + namePart2 + ' delay')
                ts.plot(rot=30,yerr = tsErr,xticks = xtickList,fmt='o', ecolor='g')
                #ts.plot(rot=30)
                #ts1.plot(rot=30)
                #ts2.plot(rot=30,xticks = xtickList)
                #plt.legend()
                plt.savefig(name2)
                plt.clf()
        
                name3 = plotFolderName + '/' + folderName + '_Queue' + nameIte + '.pdf'
                ts = pd.Series(queue95List,index = timeList)
                ts1 = pd.Series(queueMaxList,index = timeList)
                plt.figure()
                plt.ylabel('Queue length percentage (%)')
                plt.xlabel('Time')
                plt.title(namePart1 + ' ' + namePart2 + ' queue')
                ts.plot(rot=30, label = '95 Percentile')
                ts1.plot(rot=30, label = 'Max',xticks = xtickList)
                plt.legend()
                plt.savefig(name3)
                plt.clf()
                
                name4 = plotFolderName + '/' + folderName + '_SampleSize' + nameIte + '.pdf'
                ts = pd.Series(sampleList,index = timeList)
                plt.figure()
                plt.ylabel('Sample size')
                plt.xlabel('Time')
                plt.title(namePart1 + ' ' + namePart2 + ' sample size')
                ts.plot(rot=30,xticks = xtickList)
                plt.savefig(name4)
                plt.clf()
    return
    
def plotStatsInOneFolder(plotFolderName, folderName,pathname,latW,latE,latN,latS,minGroup = 10):
    '''
    This function is to calculate the mean value of delay and queuePercentageLength for different time of day and different directions. It also generates plots
    Parameters:
    -------------
    TripsDF: trips dataframe
    filenameString: the name string you want to include in the file names
    minGroup: time interval you want to use to aggregate the data. Default is 10min, which means that 1 hour will be segmented as 6 segments    

    Returns:
    -------------
    None.
    
    Outputs:
    -------------
    1 file named _stats.csv: this is the file have the mean values () of all time periods for all directions. Notice that the timeGroup variable indicates time of day. \
    If minGroup = 10, then it is calculated by hour * 6 +  min/10. So it's ranging from 0 to 143.
    A group of csv files named as (a, b)_*Time.csv: a is the startRegion, b is the endRegion. This file has information for the specified direction
        west-1
        east-2
        north-3
        south-4
    A group of pdf files named as Delay(a, b)_*Time.pdf: delay plot for the specified direction
    A group of pdf files named as Queue(a, b)_*Time.pdf: Queue plot for the specified direction. Notice the queue is the percentage of queue length to the segment length
    '''
    folderFullName = pathname + '/' + folderName
    tripsFolderName = folderFullName + '/trips'
    tempList = generateDirectionNameListFromLocation(latW,latE,latN,latS)
    if not os.path.isdir(plotFolderName):
        os.makedirs(plotFolderName)
    for nameIte in tempList:
        print 'Create plot for ', nameIte
        tripsFolderName1 = tripsFolderName + '/' + nameIte
        plotFolderName1 = plotFolderName + '/' + nameIte
        timeGroupList = []
        timeList = []
        delayMeanList = []
        delayStdList = []
        delay75List = []
        delay25List = []
        queue95List = []
        queueMaxList = []
        sampleList = []
        if os.path.isdir(tripsFolderName1):
            if len(os.listdir(tripsFolderName1)) > 0:
                for i in os.listdir(tripsFolderName1):
                    timeGroupList.append(int(i.split('.')[0]))
                    filename = tripsFolderName1 + '/' + i
                    TripsDF = pd.read_csv(filename,header = None, names = ['ID','startIndex','startDate','startTime','startLat','startLong','startSpeed','endIndex','endDate','endTime','endLat','endLong','endSpeed','startRegion','endRegion','delayCalIndex','delayCalLat','delayCalLong','delayCalTime','delay','queueIndex','queueLat','queueLong','queueTime','queueSpeed','queueLength','queuePercentageLength','intersectionIndex','intersectionLat','intersectionLong','intersectionTime','intersectionSpeed'])
                    delayMeanList.append(TripsDF['delay'].mean())
                    
                    delayStdList.append(TripsDF['delay'].std())
                    delay75List.append(TripsDF['delay'].quantile(0.75))
                    delay25List.append(TripsDF['delay'].quantile(0.25))
                    temp = TripsDF['queuePercentageLength'].quantile(0.95)*100
                    if temp < 0:
                        temp = 0
                    queue95List.append(temp)
                    temp = TripsDF['queuePercentageLength'].max()*100
                    if temp < 0:
                        temp = 0
                    queueMaxList.append(temp)
                    temp = int(i.split('.')[0])
                    minGroup = 10
                    timeList.append(time(temp/(60/minGroup),temp%(60/minGroup)*minGroup))
                    sampleList.append(TripsDF.shape[0])
            n1 = int(nameIte[0])
            n2 = int(nameIte[2])
            if n1 == 1:
                namePart1 = 'Eastbound'
                if n2 == 2:
                    namePart2 = 'through'
                elif n2 == 3:
                    namePart2 = 'left turn'
                else:
                    namePart2 = 'right turn'
            elif n1 == 2:
                namePart1 = 'Westbound'
                if n2 == 1:
                    namePart2 = 'through'
                elif n2 == 3:
                    namePart2 = 'right turn'
                else:
                    namePart2 = 'left turn'
            elif n1 == 3:
                namePart1 = 'Southbound'
                if n2 == 1:
                    namePart2 = 'right turn'
                elif n2 == 2:
                    namePart2 = 'left turn'
                else:
                    namePart2 = 'through'
            else:
                namePart1 = 'Northbound'
                if n2 == 1:
                    namePart2 = 'left turn'
                elif n2 == 2:
                    namePart2 = 'right turn'
                else:
                    namePart2 = 'through'
    
            xtickList = []
            for i in range(0,24,2):
                xtickList.append(time(i))   
            if len(delayMeanList) > 0:
                name2 = plotFolderName + '/' + folderName + '_Delay' + nameIte + '.pdf'
                ts = pd.Series(delayMeanList,index = timeList)
                #ts1 = pd.Series(delay75List,index = timeList)
                #ts2 = pd.Series(delay25List,index = timeList)
                tsErr = pd.Series(delayStdList,index = timeList)
                plt.figure()
                plt.ylabel('Delay (s)')
                plt.xlabel('Time')
                plt.title(namePart1 + ' ' + namePart2 + ' delay')
                ts.plot(rot=30,yerr = tsErr,xticks = xtickList,fmt='o', ecolor='g')
                #ts.plot(rot=30)
                #ts1.plot(rot=30)
                #ts2.plot(rot=30,xticks = xtickList)
                #plt.legend()
                plt.savefig(name2)
                plt.clf()
            

                name3 = plotFolderName + '/' + folderName + '_Queue' + nameIte + '.pdf'
                ts = pd.Series(queue95List,index = timeList)
                ts1 = pd.Series(queueMaxList,index = timeList)
                plt.figure()
                plt.ylabel('Queue length percentage (%)')
                plt.xlabel('Time')
                plt.title(namePart1 + ' ' + namePart2 + ' queue')
                ts.plot(rot=30, label = '95 Percentile')
                ts1.plot(rot=30, label = 'Max',xticks = xtickList)
                plt.legend()
                plt.savefig(name3)
                plt.clf()
            

                name4 = plotFolderName + '/' + folderName + '_SampleSize' + nameIte + '.pdf'
                ts = pd.Series(sampleList,index = timeList)
                plt.figure()
                plt.ylabel('Sample size')
                plt.xlabel('Time')
                plt.title(namePart1 + ' ' + namePart2 + ' sample size')
                ts.plot(rot=30,xticks = xtickList)
                plt.savefig(name4)
                plt.clf()
    return

def plotDelayCDF(intName, inputFolderName,outputFolderName,direction,CL,labelTime1,labelTime2):
    '''
    This function is to calculate the mean value of delay and queuePercentageLength for different time of day and different directions. It also generates plots
    Parameters:
    -------------
    TripsDF: trips dataframe
    filenameString: the name string you want to include in the file names
    minGroup: time interval you want to use to aggregate the data. Default is 10min, which means that 1 hour will be segmented as 6 segments    

    Returns:
    -------------
    None.
    
    Outputs:
    -------------
    1 file named _stats.csv: this is the file have the mean values () of all time periods for all directions. Notice that the timeGroup variable indicates time of day. \
    If minGroup = 10, then it is calculated by hour * 6 +  min/10. So it's ranging from 0 to 143.
    A group of csv files named as (a, b)_*Time.csv: a is the startRegion, b is the endRegion. This file has information for the specified direction
        west-1
        east-2
        north-3
        south-4
    A group of pdf files named as Delay(a, b)_*Time.pdf: delay plot for the specified direction
    A group of pdf files named as Queue(a, b)_*Time.pdf: Queue plot for the specified direction. Notice the queue is the percentage of queue length to the segment length
    '''
    tripsFolderName1 = inputFolderName
    AMDelayList = []
    PMDelayList = []
    if len(os.listdir(tripsFolderName1)) > 0:
        for i in os.listdir(tripsFolderName1):
            tempTimeGroup = int(i.split('.')[0])
            if peakHourCheckMinGroupAM(tempTimeGroup):
                filename = tripsFolderName1 + '/' + i
                TripsDF = pd.read_csv(filename,header = None, names = ['ID','startIndex','startDate','startTime','startLat','startLong','startSpeed','endIndex','endDate','endTime','endLat','endLong','endSpeed','startRegion','endRegion','delayCalIndex','delayCalLat','delayCalLong','delayCalTime','delay','queueIndex','queueLat','queueLong','queueTime','queueSpeed','queueLength','queuePercentageLength','intersectionIndex','intersectionLat','intersectionLong','intersectionTime','intersectionSpeed'])
                AMDelayList = AMDelayList + list(TripsDF['delay'])
            elif peakHourCheckMinGroupPM(tempTimeGroup):
                filename = tripsFolderName1 + '/' + i
                TripsDF = pd.read_csv(filename,header = None, names = ['ID','startIndex','startDate','startTime','startLat','startLong','startSpeed','endIndex','endDate','endTime','endLat','endLong','endSpeed','startRegion','endRegion','delayCalIndex','delayCalLat','delayCalLong','delayCalTime','delay','queueIndex','queueLat','queueLong','queueTime','queueSpeed','queueLength','queuePercentageLength','intersectionIndex','intersectionLat','intersectionLong','intersectionTime','intersectionSpeed'])
                PMDelayList = PMDelayList + list(TripsDF['delay'])
        n1 = int(direction[0])
        n2 = int(direction[2])
        if n1 == 1:
            namePart1 = 'Eastbound'
            if n2 == 2:
                namePart2 = 'through'
            elif n2 == 3:
                namePart2 = 'left turn'
            else:
                namePart2 = 'right turn'
        elif n1 == 2:
            namePart1 = 'Westbound'
            if n2 == 1:
                namePart2 = 'through'
            elif n2 == 3:
                namePart2 = 'right turn'
            else:
                namePart2 = 'left turn'
        elif n1 == 3:
            namePart1 = 'Southbound'
            if n2 == 1:
                namePart2 = 'right turn'
            elif n2 == 2:
                namePart2 = 'left turn'
            else:
                namePart2 = 'through'
        else:
            namePart1 = 'Northbound'
            if n2 == 1:
                namePart2 = 'left turn'
            elif n2 == 2:
                namePart2 = 'right turn'
            else:
                namePart2 = 'through'

        if len(AMDelayList) > 10:
            name2 = outputFolderName + '/' + intName + '_' + direction + '_DelayCDF_' + '_AM.pdf'
            sorted_data = np.sort(AMDelayList)
            yvals=np.arange(len(sorted_data))/float(len(sorted_data))
            plt.plot(sorted_data,yvals)
            plt.show()
            plt.ylabel('CDF')
            plt.xlabel('Delay (s)')
            plt.title(intName + ' ' + namePart1 + ' ' + namePart2 + ' AM delay' + '\n' + '(Cycle Length: ' + str(CL) + 's)')
            ax = plt.gca()
            ax.axvline(labelTime1, color='k', linestyle='--')
            ax.axvline(labelTime2, color='k', linestyle='--')
            plt.savefig(name2)
            plt.clf()
        if len(PMDelayList) > 10:
            name2 = outputFolderName + '/' + intName + '_' + direction + '_DelayCDF_' + '_PM.pdf'
            sorted_data = np.sort(PMDelayList)
            yvals=np.arange(len(sorted_data))/float(len(sorted_data))
            plt.plot(sorted_data,yvals)         
            plt.ylabel('CDF')
            plt.xlabel('Delay (s)')
            plt.title(intName + ' ' + namePart1 + ' ' + namePart2 + ' PM delay' + '\n' + '(Cycle Length: ' + str(CL) + 's)')
            ax = plt.gca()
            ax.axvline(labelTime1, color='k', linestyle='--')
            ax.axvline(labelTime2, color='k', linestyle='--')
            plt.savefig(name2)
            plt.clf()
    return

def renameFiles(path):
    os.chdir(path)
    for filename in os.listdir(path):
        print os.path.basename(filename)
        n = len(filename)-4
        os.rename(filename, filename[0:n])
    return

def CycleLengthTimeConvert(t, cl):
    return t%cl

def listToFrequency(tempList,n):
    tList = []
    for i in range(n):
        tList.append(tempList.count(i))
    return tList
    
def cycleLengthAnalysis(foldername,pathname,cyclyLength):
    folderFullName = pathname + '/' + foldername
    tripsFolderName = folderFullName + '/trips'
    outFolderName = folderFullName + '/cycleLengthAnalysis'
    tempList = generateDirectionNameList(4)
    if not os.path.isdir(outFolderName):
        os.makedirs(outFolderName)
    for nameIte in tempList:
        nameTemp = outFolderName + '/' + nameIte + '.pdf'
        print nameIte, type(nameIte)
        tripsFolderName1 = tripsFolderName + '/' + nameIte
        timeList = []
        for i in os.listdir(tripsFolderName1):
            filename = tripsFolderName1 + '/' + i
            TripsDF = pd.read_csv(filename,header = None, names = ['ID','startIndex','startDate','startTime','startLat','startLong','startSpeed','endIndex','endDate','endTime','endLat','endLong','endSpeed','startRegion','endRegion','delayCalIndex','delayCalLat','delayCalLong','delayCalTime','delay','queueIndex','queueLat','queueLong','queueTime','queueSpeed','queueLength','queuePercentageLength','intersectionIndex','intersectionLat','intersectionLong','intersectionTime','intersectionSpeed'])
            TripsDFInt = TripsDF.loc[TripsDF.intersectionIndex > 0]
            for i in TripsDFInt.index.values:
                tempTime = try_parsing_time(TripsDFInt.loc[i,'intersectionTime'])
                if 7<=tempTime.hour<=10:
                    tempSec = tempTime.hour*3600 + tempTime.minute*60 + tempTime.second
                    tempSec = CycleLengthTimeConvert(tempSec, cyclyLength)
                    timeList.append(tempSec)
        tempList = listToFrequency(timeList,cyclyLength)
        
        plt.figure()
        plt.ylabel('Counts')
        plt.xlabel('Second')
        plt.title(nameIte)
        plt.plot(tempList)
        plt.savefig(nameTemp)
    return 

def mainFunction_new(dataPath,foldername,pathname, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed,timeDiffThres):
    '''
    Main function
    Parameters:
    -------------
    namelist,pointsfilename,tripsfilename,tripscompletefilename,pathname, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed,timeDiffThres

    Returns:
    -------------
    TripsDF_Complete: trips dataframe with all information added
    '''

    filetemp = os.listdir(dataPath)[0]
    if filetemp[-1] == 'v':
        renameFiles(dataPath)
    namelist = glob.glob(dataPath + '/*.gz')
    folderFullName = pathname + '/' + foldername
    print 'Step 1:'
    if not os.path.isdir(folderFullName):
        os.makedirs(folderFullName)
    dataFolderName = folderFullName + '/' + 'filteredData'
    if not os.path.isdir(dataFolderName):
        print "Filter Data"
        dataFilterMainAllDay_new(namelist,folderFullName,latN,latS,longE,longW)
    print "Data Filtered"
    print 'Step 2:'
    tripsFolderName = folderFullName + '/' + 'trips'
    if not os.path.isdir(tripsFolderName):
        print "Create Trips"
        createTripsFile_new(tripsFolderName,dataFolderName,timeDiffThres,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
        latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed)
    print "Trips Created"
    #print 'Step 3:'
    #plotFolderName = folderFullName + '/' + 'plots'
    #if not os.path.isdir(plotFolderName):
    #    print "Create Plots"
    #    plotStats(foldername,pathname)
    #print "Plots Created"
    return
       
def mainFunction(namelist,pathname,timeDiffThres,foldername, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed):
    '''
    Main function
    Parameters:
    -------------
    namelist,pointsfilename,tripsfilename,tripscompletefilename,pathname, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed,timeDiffThres

    Returns:
    -------------
    TripsDF_Complete: trips dataframe with all information added
    '''
    folderFullName = pathname + '/' + foldername
    dataFolderName = folderFullName + '/' + 'filteredData'
    #print 'Step 1:'
    #if not os.path.isdir(folderFullName):
    #    os.makedirs(folderFullName)
    #
    #if not os.path.isdir(dataFolderName):
    #    print "Filter Data"
    #    latList = [x for x in [latW,latE,latN,latS] if x > 0]
    #    longList =  [x for x in [longW,longE,longN,longS] if x > 0]
    #    dataFilterMainAllDay(namelist,folderFullName,max(latList),min(latList),max(longList),min(longList))
    #print "Data Filtered"

    #print 'Step 2:'
    #tripsFolderName = folderFullName + '/' + 'trips'
    #if not os.path.isdir(tripsFolderName):
    #    print "Create Trips"
    #    createTripsFile(tripsFolderName,dataFolderName,timeDiffThres,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
    #    latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed)
    #print "Trips Created"

    print 'Step 3:'
    plotFolderName = folderFullName + '/' + 'plots'
    if not os.path.isdir(plotFolderName):
        print "Create Plots"
        plotStats(foldername,pathname,latW,latE,latN,latS)
    print "Plots Created"
    return

def mainFunction_step1_dataFilter_cebuNew(namelist,pathname,timeDiffThres,foldername, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed):
    '''
    Main function
    Parameters:
    -------------
    namelist,pointsfilename,tripsfilename,tripscompletefilename,pathname, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed,timeDiffThres

    Returns:
    -------------
    TripsDF_Complete: trips dataframe with all information added
    '''
    folderFullName = pathname + '/' + foldername
    print 'Step 1:', foldername
    if not os.path.isdir(folderFullName):
        os.makedirs(folderFullName)
    dataFolderName = folderFullName + '/' + 'filteredData'
    if not os.path.isdir(dataFolderName):
        print "Filter Data"
        latList = [x for x in [latW,latE,latN,latS] if x > 0]
        longList =  [x for x in [longW,longE,longN,longS] if x > 0]
        dataFilterMainAllDay_cebuNew(namelist,folderFullName,max(latList),min(latList),max(longList),min(longList))
    print "Data Filtered"
    return
    
def mainFunction_step1_dataFilter(namelist,pathname,timeDiffThres,foldername, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed):
    '''
    Main function
    Parameters:
    -------------
    namelist,pointsfilename,tripsfilename,tripscompletefilename,pathname, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed,timeDiffThres

    Returns:
    -------------
    TripsDF_Complete: trips dataframe with all information added
    '''
    folderFullName = pathname + '/' + foldername
    print 'Step 1:', foldername
    if not os.path.isdir(folderFullName):
        os.makedirs(folderFullName)
    dataFolderName = folderFullName + '/' + 'filteredData'
    if not os.path.isdir(dataFolderName):
        print "Filter Data"
        latList = [x for x in [latW,latE,latN,latS] if x > 0]
        longList =  [x for x in [longW,longE,longN,longS] if x > 0]
        dataFilterMainAllDay(namelist,folderFullName,max(latList),min(latList),max(longList),min(longList))
    print "Data Filtered"
    return

def mainFunction_step2_createTrips_cebuNew(namelist,pathname,timeDiffThres,foldername, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed):
    '''
    Main function
    Parameters:
    -------------
    namelist,pointsfilename,tripsfilename,tripscompletefilename,pathname, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed,timeDiffThres

    Returns:
    -------------
    TripsDF_Complete: trips dataframe with all information added
    '''
    folderFullName = pathname + '/' + foldername
    dataFolderName = folderFullName + '/' + 'filteredData'
    print 'Step 2:',foldername
    tripsFolderName = folderFullName + '/' + 'trips'
    if not os.path.isdir(tripsFolderName):
        print "Create Trips"
        for directories in os.listdir(dataFolderName):
            dataFolderName1 = dataFolderName + '/' + directories
            tripsFolderName1 = tripsFolderName + '/' + directories
            createTripsFile_singleCore(tripsFolderName1,dataFolderName1,timeDiffThres,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
            latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed)
    print "Trips Created"
    return

def mainFunction_step2_createTrips(namelist,pathname,timeDiffThres,foldername, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed):
    '''
    Main function
    Parameters:
    -------------
    namelist,pointsfilename,tripsfilename,tripscompletefilename,pathname, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed,timeDiffThres

    Returns:
    -------------
    TripsDF_Complete: trips dataframe with all information added
    '''
    folderFullName = pathname + '/' + foldername
    dataFolderName = folderFullName + '/' + 'filteredData'
    print 'Step 2:',foldername
    tripsFolderName = folderFullName + '/' + 'trips'
    if not os.path.isdir(tripsFolderName):
        print "Create Trips"
        createTripsFile_singleCore(tripsFolderName,dataFolderName,timeDiffThres,westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
        latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed)
    print "Trips Created"
    return

def mainFunction_step3_plotStats(namelist,pathname,timeDiffThres,foldername, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed):
    '''
    Main function
    Parameters:
    -------------
    namelist,pointsfilename,tripsfilename,tripscompletefilename,pathname, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed,timeDiffThres

    Returns:
    -------------
    TripsDF_Complete: trips dataframe with all information added
    '''
    folderFullName = pathname + '/' + foldername
    dataFolderName = folderFullName + '/' + 'filteredData'
    tripsFolderName = folderFullName + '/' + 'trips'
    print 'Step 3:',foldername
    plotFolderName = folderFullName + '/' + 'plots'
    if not os.path.isdir(plotFolderName):
        print "Create Plots"
        plotStats(foldername,pathname,latW,latE,latN,latS)
    print "Plots Created"
    return

def mainFunction_step3_plotStats_SaveInOneFolder(namelist,pathname,timeDiffThres,foldername, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed):
    '''
    Main function
    Parameters:
    -------------
    namelist,pointsfilename,tripsfilename,tripscompletefilename,pathname, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed,timeDiffThres

    Returns:
    -------------
    TripsDF_Complete: trips dataframe with all information added
    '''
    folderFullName = pathname + '/' + foldername
    dataFolderName = folderFullName + '/' + 'filteredData'
    tripsFolderName = folderFullName + '/' + 'trips'
    print 'Step 3:',foldername
    plotFolderName = pathname + '/Outputs/' + 'plots'
    plotStatsInOneFolder(plotFolderName, foldername,pathname,latW,latE,latN,latS)
    print "Plots Created"
    return

def mainFunction_step3_plotStats_SaveInOneFolder_eachSingleDay(namelist,pathname,timeDiffThres,foldername, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed):
    '''
    Main function
    Parameters:
    -------------
    namelist,pointsfilename,tripsfilename,tripscompletefilename,pathname, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed,timeDiffThres

    Returns:
    -------------
    TripsDF_Complete: trips dataframe with all information added
    '''
    folderFullName = pathname + '/' + foldername
    dataFolderName = folderFullName + '/' + 'filteredData'
    tripsFolderName = folderFullName + '/' + 'trips'
    print 'Step 3:',foldername
    plotFolderName = pathname + '/Outputs/' + 'plots'
    plotStatsInOneFolder_eachSingleDay(plotFolderName, foldername,pathname,latW,latE,latN,latS)
    print "Plots Created"
    return
    
def mainFunction_step3_plotStats_SaveInOneFolder_mergeMultipleDays(namelist,pathname,timeDiffThres,foldername, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,\
latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed):
    '''
    Main function
    Parameters:
    -------------
    namelist,pointsfilename,tripsfilename,tripscompletefilename,pathname, westcenlat, westcenlon, eastcenlat,eastcenlon, northcenlat,northcenlon,southcenlat,southcenlon,latW,longW,latE,longE,latN,longN,latS,longS,latC,longC,EWspeed,NSspeed,timeDiffThres

    Returns:
    -------------
    TripsDF_Complete: trips dataframe with all information added
    '''
    folderFullName = pathname + '/' + foldername
    dataFolderName = folderFullName + '/' + 'filteredData'
    tripsFolderName = folderFullName + '/' + 'trips'
    print 'Step 3:',foldername
    plotFolderName = pathname + '/Outputs/' + 'plots'
    plotStatsInOneFolder_mergeMultipleDays(plotFolderName, foldername,pathname,latW,latE,latN,latS)
    print "Plots Created"
    return

def mainFunction_PlotDelayCDF(pathname):
    tempFolderName = pathname + '/' + 'Inputs'
    outTempFolderName =  pathname + '/' + 'Outputs/CDFOutputs'
    if not os.path.isdir(outTempFolderName):
        os.makedirs(outTempFolderName)
    for tempFile in os.listdir(tempFolderName):
        name1 = tempFile.split('.')[0]
        print name1
        tempName = pathname + '/' +  name1 + '/trips'
        tempFileFullName = tempFolderName  + '/' + tempFile
        STDF = pd.read_csv(tempFileFullName, index_col=0)
        STDF = STDF.loc[STDF.Phase > 0]
        for i in STDF.index.values:
            if STDF.loc[i,'Phase'] >= 0:
                tempG = STDF.loc[i,'Green']
                tempCL = STDF.loc[i,'CL']
                tempR = STDF.loc[i,'Red']
                tempName1 = tempName + '/' + i
                plotDelayCDF(name1, tempName1,outTempFolderName,i,tempCL,tempR + 10,tempR + 10 + tempCL)
    return
    
def peakHourCheckMinGroupAM(i):
    return 42 <= i <= 65

def peakHourCheckMinGroupPM(i):
    return 90 <= i <= 113

def signalTiming(pathname):
    tempFolderName = pathname + '/' + 'Inputs'
    outTempFolderName =  pathname + '/' + 'Outputs'
    if not os.path.isdir(outTempFolderName):
        os.makedirs(outTempFolderName)
    singleOutFile = outTempFolderName + '/Results.csv'
    fout = open(singleOutFile, 'a')
    for tempFile in os.listdir(tempFolderName):
        name1 = tempFile.split('.')[0]
        outTempFolderName1 = outTempFolderName + '/' + name1
        if not os.path.isdir(outTempFolderName1):
            os.makedirs(outTempFolderName1)
        tempOutName = outTempFolderName1 + '/signalTimingAMPeak.csv'
        if not os.path.isfile(tempOutName):
            print 'Calculate signal timing for ', name1, ':'
            tempName = pathname + '/' +  name1 + '/trips'
            tempFileFullName = tempFolderName  + '/' + tempFile
            STDF = pd.read_csv(tempFileFullName, index_col=0)
            STDF = STDF.loc[STDF.Phase > 0]
            STDF['Q'] = None
            STDF['GtoQ'] = None
            STDF['relativeQ'] = None
            STDF['GtorelativeQ'] = None
            STDF['Delay1'] = None
            STDF['Delay2'] = None
            STDF['Delay3'] = None
            STDF['AvgDelay'] = None
            STDF['SampleSize'] = None
            for i in STDF.index.values:
                print i,':AM'
                if STDF.loc[i,'Phase'] >= 0:
                    tempG = STDF.loc[i,'Green']
                    tempCL = STDF.loc[i,'CL']
                    tempR = STDF.loc[i,'Red']
                    tempName1 = tempName + '/' + i
                    tempQueueList = []
                    tempRelativeQueueList = []
                    tempDelayList = []
                    for k in os.listdir(tempName1):
                        if peakHourCheckMinGroupAM(int(k.split('.')[0])):
                            tempName2 = tempName1 + '/' + k
                            TripsDF = pd.read_csv(tempName2,header = None, names = ['ID','startIndex','startDate','startTime','startLat','startLong','startSpeed','endIndex','endDate','endTime','endLat','endLong','endSpeed','startRegion','endRegion','delayCalIndex','delayCalLat','delayCalLong','delayCalTime','delay','queueIndex','queueLat','queueLong','queueTime','queueSpeed','queueLength','queuePercentageLength','intersectionIndex','intersectionLat','intersectionLong','intersectionTime','intersectionSpeed'])
                            temp = TripsDF['queueLength'].quantile(0.95)
                            if temp < 0:
                                temp = 0
                            tempQueueList.append(temp)
                            temp = TripsDF['queuePercentageLength'].quantile(0.95)
                            if temp < 0:
                                temp = 0
                            tempRelativeQueueList.append(temp)
                            tempDelayList = tempDelayList + list(TripsDF['delay'])
                    STDF.loc[i,'relativeQ'] = np.mean(tempRelativeQueueList)
                    STDF.loc[i,'GtorelativeQ'] = tempG / np.mean(tempRelativeQueueList)
                    STDF.loc[i,'Q'] = np.mean(tempQueueList)
                    STDF.loc[i,'GtoQ'] = tempG / np.mean(tempQueueList)
                    tempD1List = [x for x in tempDelayList if (x < (tempR + 10)) ]
                    tempD2List = [x for x in tempDelayList if (tempR + 10) < x < (tempCL + tempR + 10) ]
                    if len(tempDelayList) > 0:
                        STDF.loc[i,'Delay1'] = len(tempD1List) * 1.0 / len(tempDelayList)
                        STDF.loc[i,'Delay2'] = len(tempD2List) * 1.0 / len(tempDelayList)
                        STDF.loc[i,'Delay3'] = (len(tempDelayList) - len(tempD1List) - len(tempD2List)) * 1.0 / len(tempDelayList)
                        STDF.loc[i,'AvgDelay'] = np.mean(tempDelayList)
                        STDF.loc[i,'SampleSize'] = len(tempDelayList)
            STDF.to_csv(tempOutName)
            fout.write('\n' + name1 + '  AM Peak \n')
            for line in open(tempOutName):
                fout.write(line)

            tempOutName = outTempFolderName1 + '/signalTimingPMPeak.csv'
            tempName = pathname + '/' +  name1 + '/trips'
            STDF = pd.read_csv(tempFileFullName,index_col=0)
            STDF = STDF.loc[STDF.Phase > 0]
            STDF['Q'] = None
            STDF['GtoQ'] = None
            STDF['relativeQ'] = None
            STDF['GtorelativeQ'] = None
            STDF['Delay1'] = None
            STDF['Delay2'] = None
            STDF['Delay3'] = None
            STDF['AvgDelay'] = None
            STDF['SampleSize'] = None
            for i in STDF.index.values:
                print i,':PM'
                if STDF.loc[i,'Phase'] >= 0:
                    tempG = STDF.loc[i,'Green']
                    tempCL = STDF.loc[i,'CL']
                    tempR = STDF.loc[i,'Red']
                    tempName1 = tempName + '/' + i
                    tempQueueList = []
                    tempRelativeQueueList = []
                    tempDelayList = []
                    for k in os.listdir(tempName1):
                        if peakHourCheckMinGroupPM(int(k.split('.')[0])):
                            tempName2 = tempName1 + '/' + k
                            TripsDF = pd.read_csv(tempName2,header = None, names = ['ID','startIndex','startDate','startTime','startLat','startLong','startSpeed','endIndex','endDate','endTime','endLat','endLong','endSpeed','startRegion','endRegion','delayCalIndex','delayCalLat','delayCalLong','delayCalTime','delay','queueIndex','queueLat','queueLong','queueTime','queueSpeed','queueLength','queuePercentageLength','intersectionIndex','intersectionLat','intersectionLong','intersectionTime','intersectionSpeed'])
                            temp = TripsDF['queueLength'].quantile(0.95)
                            if temp < 0:
                                temp = 0
                            tempQueueList.append(temp)
                            temp = TripsDF['queuePercentageLength'].quantile(0.95)
                            if temp < 0:
                                temp = 0
                            tempRelativeQueueList.append(temp)
                            tempDelayList = tempDelayList + list(TripsDF['delay'])
                    STDF.loc[i,'relativeQ'] = np.mean(tempRelativeQueueList)
                    STDF.loc[i,'GtorelativeQ'] = tempG / np.mean(tempRelativeQueueList)
                    STDF.loc[i,'Q'] = np.mean(tempQueueList)
                    STDF.loc[i,'GtoQ'] = tempG / np.mean(tempQueueList)
                    tempD1List = [x for x in tempDelayList if (x < (tempR + 10)) ]
                    tempD2List = [x for x in tempDelayList if (tempR + 10) < x < (tempCL + tempR + 10) ]
                    if len(tempDelayList) > 0:
                        STDF.loc[i,'Delay1'] = len(tempD1List) * 1.0 / len(tempDelayList)
                        STDF.loc[i,'Delay2'] = len(tempD2List) * 1.0 / len(tempDelayList)
                        STDF.loc[i,'Delay3'] = (len(tempDelayList) - len(tempD1List) - len(tempD2List)) * 1.0 / len(tempDelayList)
                        STDF.loc[i,'AvgDelay'] = np.mean(tempDelayList)
                        STDF.loc[i,'SampleSize'] = len(tempDelayList)
            STDF.to_csv(tempOutName)
            fout.write('\n' + name1 + '   PM Peak \n')
            for line in open(tempOutName):
                fout.write(line)
    fout.close()
    return
    
def signalTiming_AMPM(pathname):
    tempFolderName = pathname + '/' + 'Inputs'
    outTempFolderName =  pathname + '/' + 'Outputs'
    if not os.path.isdir(outTempFolderName):
        os.makedirs(outTempFolderName)
    singleOutFile = outTempFolderName + '/Results.csv'
    fout = open(singleOutFile, 'a')
    for tempFile in os.listdir(tempFolderName):
        name1 = tempFile.split('.')[0]
        print name1
        name2_intersectionName = name1.split('_')[0]
        name2_ampm = name1.split('_')[1]
        outTempFolderName1 = outTempFolderName + '/' + name2_intersectionName
        if not os.path.isdir(outTempFolderName1):
            os.makedirs(outTempFolderName1)
        if name2_ampm == 'am':
            tempOutName = outTempFolderName1 + '/signalTimingAMPeak.csv'
            if not os.path.isfile(tempOutName):
                print 'Calculate signal timing for ', name1, ':'
                tempName = pathname + '/' +  name2_intersectionName + '/trips'
                tempFileFullName = tempFolderName  + '/' + tempFile
                STDF = pd.read_csv(tempFileFullName, index_col=0)
                STDF = STDF.loc[STDF.Phase > 0]
                STDF['Q'] = None
                STDF['GtoQ'] = None
                STDF['relativeQ'] = None
                STDF['GtorelativeQ'] = None
                STDF['Delay1'] = None
                STDF['Delay2'] = None
                STDF['Delay3'] = None
                STDF['AvgDelay'] = None
                STDF['SampleSize'] = None
                for i in STDF.index.values:
                    print i,':AM'
                    if STDF.loc[i,'Phase'] >= 0:
                        tempG = STDF.loc[i,'Green']
                        tempCL = STDF.loc[i,'CL']
                        tempR = STDF.loc[i,'Red']
                        tempName1 = tempName + '/' + i
                        tempQueueList = []
                        tempRelativeQueueList = []
                        tempDelayList = []
                        for k in os.listdir(tempName1):
                            if peakHourCheckMinGroupAM(int(k.split('.')[0])):
                                tempName2 = tempName1 + '/' + k
                                TripsDF = pd.read_csv(tempName2,header = None, names = ['ID','startIndex','startDate','startTime','startLat','startLong','startSpeed','endIndex','endDate','endTime','endLat','endLong','endSpeed','startRegion','endRegion','delayCalIndex','delayCalLat','delayCalLong','delayCalTime','delay','queueIndex','queueLat','queueLong','queueTime','queueSpeed','queueLength','queuePercentageLength','intersectionIndex','intersectionLat','intersectionLong','intersectionTime','intersectionSpeed'])
                                temp = TripsDF['queueLength'].quantile(0.95)
                                if temp < 0:
                                    temp = 0
                                tempQueueList.append(temp)
                                temp = TripsDF['queuePercentageLength'].quantile(0.95)
                                if temp < 0:
                                    temp = 0
                                tempRelativeQueueList.append(temp)
                                tempDelayList = tempDelayList + list(TripsDF['delay'])
                        STDF.loc[i,'relativeQ'] = np.mean(tempRelativeQueueList)
                        STDF.loc[i,'GtorelativeQ'] = tempG / np.mean(tempRelativeQueueList)
                        STDF.loc[i,'Q'] = np.mean(tempQueueList)
                        STDF.loc[i,'GtoQ'] = tempG / np.mean(tempQueueList)
                        tempD1List = [x for x in tempDelayList if (x < (tempR + 10)) ]
                        tempD2List = [x for x in tempDelayList if (tempR + 10) < x < (tempCL + tempR + 10) ]
                        if len(tempDelayList) > 0:
                            STDF.loc[i,'Delay1'] = len(tempD1List) * 1.0 / len(tempDelayList)
                            STDF.loc[i,'Delay2'] = len(tempD2List) * 1.0 / len(tempDelayList)
                            STDF.loc[i,'Delay3'] = (len(tempDelayList) - len(tempD1List) - len(tempD2List)) * 1.0 / len(tempDelayList)
                            STDF.loc[i,'AvgDelay'] = np.mean(tempDelayList)
                            STDF.loc[i,'SampleSize'] = len(tempDelayList)
                STDF.to_csv(tempOutName)
                fout.write('\n' + name2_intersectionName + '  AM Peak \n')
                for line in open(tempOutName):
                    fout.write(line)
        else:
            tempOutName = outTempFolderName1 + '/signalTimingPMPeak.csv'
            tempName = pathname + '/' +  name2_intersectionName + '/trips'
            tempFileFullName = tempFolderName  + '/' + tempFile
            STDF = pd.read_csv(tempFileFullName,index_col=0)
            STDF = STDF.loc[STDF.Phase > 0]
            STDF['Q'] = None
            STDF['GtoQ'] = None
            STDF['relativeQ'] = None
            STDF['GtorelativeQ'] = None
            STDF['Delay1'] = None
            STDF['Delay2'] = None
            STDF['Delay3'] = None
            STDF['AvgDelay'] = None
            STDF['SampleSize'] = None
            for i in STDF.index.values:
                print i,':PM'
                if STDF.loc[i,'Phase'] >= 0:
                    tempG = STDF.loc[i,'Green']
                    tempCL = STDF.loc[i,'CL']
                    tempR = STDF.loc[i,'Red']
                    tempName1 = tempName + '/' + i
                    tempQueueList = []
                    tempRelativeQueueList = []
                    tempDelayList = []
                    for k in os.listdir(tempName1):
                        if peakHourCheckMinGroupPM(int(k.split('.')[0])):
                            tempName2 = tempName1 + '/' + k
                            TripsDF = pd.read_csv(tempName2,header = None, names = ['ID','startIndex','startDate','startTime','startLat','startLong','startSpeed','endIndex','endDate','endTime','endLat','endLong','endSpeed','startRegion','endRegion','delayCalIndex','delayCalLat','delayCalLong','delayCalTime','delay','queueIndex','queueLat','queueLong','queueTime','queueSpeed','queueLength','queuePercentageLength','intersectionIndex','intersectionLat','intersectionLong','intersectionTime','intersectionSpeed'])
                            temp = TripsDF['queueLength'].quantile(0.95)
                            if temp < 0:
                                temp = 0
                            tempQueueList.append(temp)
                            temp = TripsDF['queuePercentageLength'].quantile(0.95)
                            if temp < 0:
                                temp = 0
                            tempRelativeQueueList.append(temp)
                            tempDelayList = tempDelayList + list(TripsDF['delay'])
                    STDF.loc[i,'relativeQ'] = np.mean(tempRelativeQueueList)
                    STDF.loc[i,'GtorelativeQ'] = tempG / np.mean(tempRelativeQueueList)
                    STDF.loc[i,'Q'] = np.mean(tempQueueList)
                    STDF.loc[i,'GtoQ'] = tempG / np.mean(tempQueueList)
                    tempD1List = [x for x in tempDelayList if (x < (tempR + 10)) ]
                    tempD2List = [x for x in tempDelayList if (tempR + 10) < x < (tempCL + tempR + 10) ]
                    if len(tempDelayList) > 0:
                        STDF.loc[i,'Delay1'] = len(tempD1List) * 1.0 / len(tempDelayList)
                        STDF.loc[i,'Delay2'] = len(tempD2List) * 1.0 / len(tempDelayList)
                        STDF.loc[i,'Delay3'] = (len(tempDelayList) - len(tempD1List) - len(tempD2List)) * 1.0 / len(tempDelayList)
                        STDF.loc[i,'AvgDelay'] = np.mean(tempDelayList)
                        STDF.loc[i,'SampleSize'] = len(tempDelayList)
            STDF.to_csv(tempOutName)
            fout.write('\n' + name2_intersectionName + '   PM Peak \n')
            for line in open(tempOutName):
                fout.write(line)
    fout.close()
    return

    
def signalTiming_eachSingleDay(pathname):
    tempFolderName = pathname + '/' + 'Inputs'
    outTempFolderName =  pathname + '/' + 'Outputs'
    if not os.path.isdir(outTempFolderName):
        os.makedirs(outTempFolderName)
    singleOutFile = outTempFolderName + '/Results.csv'
    fout = open(singleOutFile, 'a')
    for tempFile in os.listdir(tempFolderName):
        name1 = tempFile.split('.')[0]
        outTempFolderName1 = outTempFolderName + '/' + name1
        if not os.path.isdir(outTempFolderName1):
            os.makedirs(outTempFolderName1)
        tempName = pathname + '/' +  name1 + '/trips'
        for dateName in os.listdir(tempName):
            outTempFolderName2 = outTempFolderName1 + '/' + dateName
            if not os.path.isdir(outTempFolderName2):
                os.makedirs(outTempFolderName2)
            tempOutName = outTempFolderName2 + '/signalTimingAMPeak.csv'
            if not os.path.isfile(tempOutName):
                print 'Calculate signal timing for ', name1, ':'
                tempName1 = tempName + '/' + dateName
                tempFileFullName = tempFolderName  + '/' + tempFile
                STDF = pd.read_csv(tempFileFullName, index_col=0)
                STDF = STDF.loc[STDF.Phase > 0]
                STDF['Q'] = None
                STDF['GtoQ'] = None
                STDF['relativeQ'] = None
                STDF['GtorelativeQ'] = None
                STDF['Delay1'] = None
                STDF['Delay2'] = None
                STDF['Delay3'] = None
                STDF['AvgDelay'] = None
                STDF['SampleSize'] = None
                for i in STDF.index.values:
                    print dateName, i,':AM'
                    if STDF.loc[i,'Phase'] >= 0:
                        tempG = STDF.loc[i,'Green']
                        tempCL = STDF.loc[i,'CL']
                        tempR = STDF.loc[i,'Red']
                        tempName2 = tempName1 + '/' + i
                        tempQueueList = []
                        tempRelativeQueueList = []
                        tempDelayList = []
                        for k in os.listdir(tempName2):
                            if peakHourCheckMinGroupAM(int(k.split('.')[0])):
                                tempName3 = tempName2 + '/' + k
                                TripsDF = pd.read_csv(tempName3,header = None, names = ['ID','startIndex','startDate','startTime','startLat','startLong','startSpeed','endIndex','endDate','endTime','endLat','endLong','endSpeed','startRegion','endRegion','delayCalIndex','delayCalLat','delayCalLong','delayCalTime','delay','queueIndex','queueLat','queueLong','queueTime','queueSpeed','queueLength','queuePercentageLength','intersectionIndex','intersectionLat','intersectionLong','intersectionTime','intersectionSpeed'])
                                temp = TripsDF['queueLength'].quantile(0.95)
                                if temp < 0:
                                    temp = 0
                                tempQueueList.append(temp)
                                temp = TripsDF['queuePercentageLength'].quantile(0.95)
                                if temp < 0:
                                    temp = 0
                                tempRelativeQueueList.append(temp)
                                tempDelayList = tempDelayList + list(TripsDF['delay'])
                        STDF.loc[i,'relativeQ'] = np.mean(tempRelativeQueueList)
                        STDF.loc[i,'GtorelativeQ'] = tempG / np.mean(tempRelativeQueueList)
                        STDF.loc[i,'Q'] = np.mean(tempQueueList)
                        STDF.loc[i,'GtoQ'] = tempG / np.mean(tempQueueList)
                        tempD1List = [x for x in tempDelayList if (x < (tempR + 10)) ]
                        tempD2List = [x for x in tempDelayList if (tempR + 10) < x < (tempCL + tempR + 10) ]
                        if len(tempDelayList) > 0:
                            STDF.loc[i,'Delay1'] = len(tempD1List) * 1.0 / len(tempDelayList)
                            STDF.loc[i,'Delay2'] = len(tempD2List) * 1.0 / len(tempDelayList)
                            STDF.loc[i,'Delay3'] = (len(tempDelayList) - len(tempD1List) - len(tempD2List)) * 1.0 / len(tempDelayList)
                            STDF.loc[i,'AvgDelay'] = np.mean(tempDelayList)
                            STDF.loc[i,'SampleSize'] = len(tempDelayList)
                STDF.to_csv(tempOutName)
                fout.write('\n' + name1 + ' ' + dateName + '  AM Peak \n')
                for line in open(tempOutName):
                    fout.write(line)
    
                tempOutName = outTempFolderName2 + '/signalTimingPMPeak.csv'
                tempName1 = tempName + '/' + dateName
                STDF = pd.read_csv(tempFileFullName,index_col=0)
                STDF = STDF.loc[STDF.Phase > 0]
                STDF['Q'] = None
                STDF['GtoQ'] = None
                STDF['relativeQ'] = None
                STDF['GtorelativeQ'] = None
                STDF['Delay1'] = None
                STDF['Delay2'] = None
                STDF['Delay3'] = None
                STDF['AvgDelay'] = None
                STDF['SampleSize'] = None
                for i in STDF.index.values:
                    print dateName, i,':PM'
                    if STDF.loc[i,'Phase'] >= 0:
                        tempG = STDF.loc[i,'Green']
                        tempCL = STDF.loc[i,'CL']
                        tempR = STDF.loc[i,'Red']
                        tempName2 = tempName1 + '/' + i
                        tempQueueList = []
                        tempRelativeQueueList = []
                        tempDelayList = []
                        for k in os.listdir(tempName2):
                            if peakHourCheckMinGroupPM(int(k.split('.')[0])):
                                tempName3 = tempName2 + '/' + k
                                TripsDF = pd.read_csv(tempName3,header = None, names = ['ID','startIndex','startDate','startTime','startLat','startLong','startSpeed','endIndex','endDate','endTime','endLat','endLong','endSpeed','startRegion','endRegion','delayCalIndex','delayCalLat','delayCalLong','delayCalTime','delay','queueIndex','queueLat','queueLong','queueTime','queueSpeed','queueLength','queuePercentageLength','intersectionIndex','intersectionLat','intersectionLong','intersectionTime','intersectionSpeed'])
                                temp = TripsDF['queueLength'].quantile(0.95)
                                if temp < 0:
                                    temp = 0
                                tempQueueList.append(temp)
                                temp = TripsDF['queuePercentageLength'].quantile(0.95)
                                if temp < 0:
                                    temp = 0
                                tempRelativeQueueList.append(temp)
                                tempDelayList = tempDelayList + list(TripsDF['delay'])
                        STDF.loc[i,'relativeQ'] = np.mean(tempRelativeQueueList)
                        STDF.loc[i,'GtorelativeQ'] = tempG / np.mean(tempRelativeQueueList)
                        STDF.loc[i,'Q'] = np.mean(tempQueueList)
                        STDF.loc[i,'GtoQ'] = tempG / np.mean(tempQueueList)
                        tempD1List = [x for x in tempDelayList if (x < (tempR + 10)) ]
                        tempD2List = [x for x in tempDelayList if (tempR + 10) < x < (tempCL + tempR + 10) ]
                        if len(tempDelayList) > 0:
                            STDF.loc[i,'Delay1'] = len(tempD1List) * 1.0 / len(tempDelayList)
                            STDF.loc[i,'Delay2'] = len(tempD2List) * 1.0 / len(tempDelayList)
                            STDF.loc[i,'Delay3'] = (len(tempDelayList) - len(tempD1List) - len(tempD2List)) * 1.0 / len(tempDelayList)
                            STDF.loc[i,'AvgDelay'] = np.mean(tempDelayList)
                            STDF.loc[i,'SampleSize'] = len(tempDelayList)
                STDF.to_csv(tempOutName)
                fout.write('\n' +  name1 + ' ' + dateName  +'   PM Peak \n')
                for line in open(tempOutName):
                    fout.write(line)
    fout.close()
    return

def peakHourCheckMinGroupAMHour(peakHour,i):
    return (peakHour + 7)*6 <= i < (peakHour + 8)*6

def peakHourCheckMinGroupPMHour(peakHour,i):
    return (peakHour + 15)*6 <= i < (peakHour + 16)*6
    
def signalTiming_EachHour(pathname):
    tempFolderName = pathname + '/' + 'Inputs'
    outTempFolderName =  pathname + '/' + 'Outputs'
    if not os.path.isdir(outTempFolderName):
        os.makedirs(outTempFolderName)
    singleOutFile = outTempFolderName + '/ResultsForEachHour.csv'
    fout = open(singleOutFile, 'a')
    for tempFile in os.listdir(tempFolderName):
        name1 = tempFile.split('.')[0]
        print name1
        name2_intersectionName = name1.split('_')[0]
        name2_ampm = name1.split('_')[1]
        outTempFolderName1 = outTempFolderName + '/' + name2_intersectionName
        if not os.path.isdir(outTempFolderName1):
            os.makedirs(outTempFolderName1)
        if name2_ampm == 'am':
            for eachPeakHour in range(4):
                tempOutName = outTempFolderName1 + '/signalTimingAMPeak' + str(eachPeakHour) + '.csv'
                if not os.path.isfile(tempOutName):
                    print 'Calculate signal timing for ', name1, ':'
                    tempName = pathname + '/' +  name2_intersectionName + '/trips'
                    tempFileFullName = tempFolderName  + '/' + tempFile
                    STDF = pd.read_csv(tempFileFullName, index_col=0)
                    STDF = STDF.loc[STDF.Phase > 0]
                    STDF['Q'] = None
                    STDF['GtoQ'] = None
                    STDF['relativeQ'] = None
                    STDF['GtorelativeQ'] = None
                    STDF['Delay1'] = None
                    STDF['Delay2'] = None
                    STDF['Delay3'] = None
                    STDF['AvgDelay'] = None
                    STDF['SampleSize'] = None
                    for i in STDF.index.values:
                        print i,':AM'
                        if STDF.loc[i,'Phase'] >= 0:
                            tempG = STDF.loc[i,'Green']
                            tempCL = STDF.loc[i,'CL']
                            tempR = STDF.loc[i,'Red']
                            tempName1 = tempName + '/' + i
                            tempQueueList = []
                            tempRelativeQueueList = []
                            tempDelayList = []
                            for k in os.listdir(tempName1):
                                if peakHourCheckMinGroupAMHour(eachPeakHour, int(k.split('.')[0])):
                                    tempName2 = tempName1 + '/' + k
                                    TripsDF = pd.read_csv(tempName2,header = None, names = ['ID','startIndex','startDate','startTime','startLat','startLong','startSpeed','endIndex','endDate','endTime','endLat','endLong','endSpeed','startRegion','endRegion','delayCalIndex','delayCalLat','delayCalLong','delayCalTime','delay','queueIndex','queueLat','queueLong','queueTime','queueSpeed','queueLength','queuePercentageLength','intersectionIndex','intersectionLat','intersectionLong','intersectionTime','intersectionSpeed'])
                                    temp = TripsDF['queueLength'].quantile(0.95)
                                    if temp < 0:
                                        temp = 0
                                    tempQueueList.append(temp)
                                    temp = TripsDF['queuePercentageLength'].quantile(0.95)
                                    if temp < 0:
                                        temp = 0
                                    tempRelativeQueueList.append(temp)
                                    tempDelayList = tempDelayList + list(TripsDF['delay'])
                            STDF.loc[i,'relativeQ'] = np.mean(tempRelativeQueueList)
                            STDF.loc[i,'GtorelativeQ'] = tempG / np.mean(tempRelativeQueueList)
                            STDF.loc[i,'Q'] = np.mean(tempQueueList)
                            STDF.loc[i,'GtoQ'] = tempG / np.mean(tempQueueList)
                            tempD1List = [x for x in tempDelayList if (x < (tempR + 10)) ]
                            tempD2List = [x for x in tempDelayList if (tempR + 10) < x < (tempCL + tempR + 10) ]
                            if len(tempDelayList) > 0:
                                STDF.loc[i,'Delay1'] = len(tempD1List) * 1.0 / len(tempDelayList)
                                STDF.loc[i,'Delay2'] = len(tempD2List) * 1.0 / len(tempDelayList)
                                STDF.loc[i,'Delay3'] = (len(tempDelayList) - len(tempD1List) - len(tempD2List)) * 1.0 / len(tempDelayList)
                                STDF.loc[i,'AvgDelay'] = np.mean(tempDelayList)
                                STDF.loc[i,'SampleSize'] = len(tempDelayList)
                    STDF.to_csv(tempOutName)
                    fout.write('\n' + name2_intersectionName + ' ' + str(eachPeakHour + 7) + ' AM\n')
                    for line in open(tempOutName):
                        fout.write(line)
        else:
            for eachPeakHour in range(4):
                tempOutName = outTempFolderName1 + '/signalTimingPMPeak' + str(eachPeakHour) + '.csv'
                tempName = pathname + '/' +  name2_intersectionName + '/trips'
                tempFileFullName = tempFolderName  + '/' + tempFile
                STDF = pd.read_csv(tempFileFullName,index_col=0)
                STDF = STDF.loc[STDF.Phase > 0]
                STDF['Q'] = None
                STDF['GtoQ'] = None
                STDF['relativeQ'] = None
                STDF['GtorelativeQ'] = None
                STDF['Delay1'] = None
                STDF['Delay2'] = None
                STDF['Delay3'] = None
                STDF['AvgDelay'] = None
                STDF['SampleSize'] = None
                for i in STDF.index.values:
                    print i,':PM'
                    if STDF.loc[i,'Phase'] >= 0:
                        tempG = STDF.loc[i,'Green']
                        tempCL = STDF.loc[i,'CL']
                        tempR = STDF.loc[i,'Red']
                        tempName1 = tempName + '/' + i
                        tempQueueList = []
                        tempRelativeQueueList = []
                        tempDelayList = []
                        for k in os.listdir(tempName1):
                            if peakHourCheckMinGroupPMHour(eachPeakHour,int(k.split('.')[0])):
                                tempName2 = tempName1 + '/' + k
                                TripsDF = pd.read_csv(tempName2,header = None, names = ['ID','startIndex','startDate','startTime','startLat','startLong','startSpeed','endIndex','endDate','endTime','endLat','endLong','endSpeed','startRegion','endRegion','delayCalIndex','delayCalLat','delayCalLong','delayCalTime','delay','queueIndex','queueLat','queueLong','queueTime','queueSpeed','queueLength','queuePercentageLength','intersectionIndex','intersectionLat','intersectionLong','intersectionTime','intersectionSpeed'])
                                temp = TripsDF['queueLength'].quantile(0.95)
                                if temp < 0:
                                    temp = 0
                                tempQueueList.append(temp)
                                temp = TripsDF['queuePercentageLength'].quantile(0.95)
                                if temp < 0:
                                    temp = 0
                                tempRelativeQueueList.append(temp)
                                tempDelayList = tempDelayList + list(TripsDF['delay'])
                        STDF.loc[i,'relativeQ'] = np.mean(tempRelativeQueueList)
                        STDF.loc[i,'GtorelativeQ'] = tempG / np.mean(tempRelativeQueueList)
                        STDF.loc[i,'Q'] = np.mean(tempQueueList)
                        STDF.loc[i,'GtoQ'] = tempG / np.mean(tempQueueList)
                        tempD1List = [x for x in tempDelayList if (x < (tempR + 10)) ]
                        tempD2List = [x for x in tempDelayList if (tempR + 10) < x < (tempCL + tempR + 10) ]
                        if len(tempDelayList) > 0:
                            STDF.loc[i,'Delay1'] = len(tempD1List) * 1.0 / len(tempDelayList)
                            STDF.loc[i,'Delay2'] = len(tempD2List) * 1.0 / len(tempDelayList)
                            STDF.loc[i,'Delay3'] = (len(tempDelayList) - len(tempD1List) - len(tempD2List)) * 1.0 / len(tempDelayList)
                            STDF.loc[i,'AvgDelay'] = np.mean(tempDelayList)
                            STDF.loc[i,'SampleSize'] = len(tempDelayList)
                STDF.to_csv(tempOutName)
                fout.write('\n' + name2_intersectionName + ' ' + str(eachPeakHour + 15) + ' PM\n')
                for line in open(tempOutName):
                    fout.write(line)
    fout.close()
    return
    
def removeNotPeakForSingleDay(pathname1):
    tempFolderName = pathname1 + '/' + 'Inputs'
    for tempFile in os.listdir(tempFolderName):
        name1 = tempFile.split('.')[0]
        tempName = pathname1 + '/' +  name1 + '/trips/2016_02_23'
        for i in os.listdir(tempName):
            tempName1 = tempName + '/' + i
            for k in os.listdir(tempName1):
                if not peakHourCheckMinGroupPM(int(k.split('.')[0])):
                    os.remove(tempName1 + '/' + k)

def mergeTripsFromMultiDays(pathname1,copypathname):
    if not os.path.isdir(copypathname):
        os.makedirs(copypathname)
    tempFolderName = pathname1 + '/' + 'Inputs'
    for tempFile in os.listdir(tempFolderName):
        name1 = tempFile.split('.')[0]
        tempFileFullName = tempFolderName  + '/' + tempFile
        mergeFolderName1 = copypathname + '/' + name1 + '/trips'
        if not os.path.isdir(mergeFolderName1):
            os.makedirs(mergeFolderName1)
        STDF = pd.read_csv(tempFileFullName, index_col=0)
        tempName = pathname1 + '/' +  name1 + '/trips'
        for i in STDF.index.values:
            if STDF.loc[i,'Phase'] >= 0:
                mergeFolderName2 = mergeFolderName1 + '/' + i
                if not os.path.isdir(mergeFolderName2):
                    os.makedirs(mergeFolderName2)
                for dateName in os.listdir(tempName):
                    tempName2 = tempName + '/' + dateName + '/' + i
                    for filename in os.listdir(tempName2):
                        tempName3 = tempName2 + '/' + filename
                        mergeOutName = mergeFolderName2 + '/' + filename
                        with open(mergeOutName, 'a') as outfile:
                            with open(tempName3) as infile:
                                outfile.write(infile.read())
                                


def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise
        
def addDataFromOneDirectoryToAnotherDirectory(pathname2,combinedPathName):
    intersectionFileName = combinedPathName + '/intersection_information.csv'
    intersectionDF = pd.read_csv(intersectionFileName,index_col=0)
    for i in intersectionDF.index.values:
        intersectionName = intersectionDF.loc[i,'folderName']
        combinedIntersectionPathName = combinedPathName + '/' + intersectionName
        if os.path.isdir(combinedIntersectionPathName):
            combinedIntersectionTripPathName = combinedIntersectionPathName + '/trips'
            addIntersectionTripPathName = pathname2 + '/' + intersectionName + '/trips'
            for directionName in os.listdir(addIntersectionTripPathName):
                addIntersectionDiretionPathName = addIntersectionTripPathName + '/' + directionName
                combineIntersectionDiretionPathName = combinedIntersectionTripPathName + '/' + directionName
                if directionName not in os.listdir(combinedIntersectionTripPathName):
                    copyanything(addIntersectionDiretionPathName, combineIntersectionDiretionPathName)
                else:
                    for timeSlot in os.listdir(addIntersectionDiretionPathName):
                        addIntersectionTimePathName = addIntersectionDiretionPathName + '/' + timeSlot
                        combineIntersectionTimePathName = combineIntersectionDiretionPathName + '/' + timeSlot
                        #if timeSlot not in os.listdir(combineIntersectionDiretionPathName):
                        #    copyanything(addIntersectionTimePathName, combineIntersectionTimePathName)
                        #else:
                        with open(combineIntersectionTimePathName, 'a') as outfile:
                            with open(addIntersectionTimePathName) as infile:
                                outfile.write(infile.read())