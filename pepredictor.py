# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 18:45:52 2014

@author: Viju
"""
import csv
import copy
from math import sqrt
from datetime import datetime


class PePredictor:
        
    def __init__(self, filename):
        self.lineCount = 0
        self.filename = filename
        self.rowHeader = []
        self.colHeader = []
        self.normalizedTickerDict = {}                          #non-normalized original global dictionary
        self.nonNormalizedTickerDict = {}                       #normalized global dictionary
        self.priceDict = {}                                     #store all the market prices in this global dictionary

    #loadTickerDict Reads a CSV file of Ticker Data and return a ticker dictionary of the following form
    #Currently loads from CSV placed in the same directory.This file contains S&P > 5Billion market cap
    #ToDo: Modify to use Panda Dataframes
    """{'AMTD': {'Forward P/E': '19.86', 'Return on Equity': '16.70%', 'Float Short': '3.96%', 
                'Insider Transactions': '-7.34%', 'Dividend Yield': '1.50%', 'P/S': '5.80', 
                'Performance (Half Year)': '7.56%', 'Insider Ownership': '9.80%', 'Current Ratio': '', 
                'Sales growth past 5 years': '-0.10%', 'Shares Float': '259.94', 'PEG': '1.14', 
                'P/B': '3.75', 'LT Debt/Equity': '0.22', 'P/E': '22.86', 'EPS growth next 5 years': '19.98%',
                'Operating Margin': '40.30%', 'Company': 'TD Ameritrade Holding Corporation', 
                'Performance (Month)': '-0.93%', 'Beta': '1.42', 'Payout Ratio': '32.10%', 
                'Sales growth quarter over quarter': '5.10%', 'EPS growth this year': '15.10%', 
                'Short Ratio': '3.9', 'P/Free Cash Flow': '', 'Ticker': 'AMTD', 
                'Performance (Quarter)': '1.11%', 'Institutional Ownership': '47.40%', 'Country': 'USA', 
                'Industry': 'Investment Brokerage - National', 'Return on Assets': '3.40%', 
                'EPS (ttm)': '1.4', 'Performance (YTD)': '5.68%', 'Performance (Year)': '20.34%', 
                'EPS growth next year': '15.80%', 'Volatility (Week)': '2.00%', 'Average True Range': '0.6',
                'Sector': 'Financial', 'Institutional Transactions': '-0.94%', 
                'Shares Outstanding': '551.57', 'Quick Ratio': '', 
                'EPS growth quarter over quarter': '3.00%', 'Total Debt/Equity': '0.25', 
                'Return on Investment': '11.20%', 'Performance (Week)': '2.20%', 
                'Volatility (Month)': '1.80%', 'P/Cash': '14.01', 'Gross Margin': '99.80%', 
                'Profit Margin': '25.50%', 'Relative Strength Index (14)': '57.94', 
                'EPS growth past 5 years': '-1.70%', 'Market Cap': '17655.75'}}
    """
    def loadTickerDict(self, filename = ''):        
        tickerDict = {}        
        if(filename == ''):
            filename = self.filename
            
        f = open(filename)
        lines = f.readlines()
        self.lineCount = len(lines)
        
        with open(filename) as csvfile:
            f_csv = csv.reader(csvfile)
            #Skipping first row after getting headers
            headers = next(f_csv)
            for row in f_csv:
                #Zipping the header with every row in the data and then pushing it inside tickerDict
                #with the tickers as keys
                tickerDict[row[0]] = dict(zip(headers,row))
        self.rowHeader = sorted(tickerDict.keys())
        self.colHeader = sorted(headers)
        #print tickerDict
        self.nonNormalizedTickerDict = copy.deepcopy(tickerDict)
        return tickerDict

    #Take a list of Column Headers and return the formatted column in a tickerDict
    #Splits the Column Header = Price into a different price dictionary object and deletes the Price Column
    #Empty Strings = '' are assigned as a None
    def formatColumn(self, colHeaderList, tickerDict):
        priceDict = {}
        for colHeader in colHeaderList:
            for ticker in tickerDict:
                if colHeader == 'Price':
                    #Assign Price into the priceDict against the ticker value
                    priceDict[ticker] = tickerDict[ticker][colHeader].strip()
                else:
                    tmpColVal = tickerDict[ticker][colHeader].strip()
                    
                if tmpColVal == '':
                    tickerDict[ticker][colHeader] = None
                elif (not self.isNumber(tmpColVal)) and ('%' not in tmpColVal):
                    tickerDict[ticker][colHeader] = tmpColVal
                elif '%' in tmpColVal:
                    tickerDict[ticker][colHeader] = float(tmpColVal[:-1])
                elif self.isNumber(tmpColVal):
                    tickerDict[ticker][colHeader] = float(tmpColVal)
                else:
                    tickerDict[ticker][colHeader] = 0.00
        #Now Delete thr price attributed from tickerDict and also from colHeaderList
        del tickerDict[ticker]["Price"]
        colHeaderList.remove("Price")
        self.priceDict = priceDict
        return tickerDict

    #Take a list of Column Headers and return the Normalized column in a tickerDict
    #None values are ignored
    def normalizeColumn(self, colHeaderList, tickerDict):
       """given a column name, normalize that column in tickerDict
       using the Modified Standard Score"""
       for colHeader in colHeaderList:
           #extract values of column Header to list and ignore None values in the column Header
           col = [tickerDict[i][colHeader] for i in tickerDict if tickerDict[i][colHeader] is not None]
           if type(col[0]) is not str:
               median = self.getMedian(col)
               #print("Median of "+ colHeader +" column:" + str(median))
               asd = self.getAbsoluteStandardDeviation(col,median)
               #print("Absolute Deviation of "+ colHeader +" column:" + str(asd))
               for v in tickerDict:
                   if tickerDict[v][colHeader] is not None:
                       tickerDict[v][colHeader] = (tickerDict[v][colHeader] - median) / asd
       self.normalizedTickerDict = tickerDict 
       return tickerDict

    #Computes the Manhattan distance between two vectors in dictionary form with numeric values in it.
    #Return the Manhattan distance between two vectors
    def manhattan(self, vector1, vector2):
        manhattanDistance = 0
        for colHeader in self.colHeader:
            if (type(vector1[colHeader])  is not str) and (vector1[colHeader]  is not None):
                if (type(vector2[colHeader])  is not str) and (vector2[colHeader]  is not None):
                    manhattanDistance += abs(vector1[colHeader] - vector2[colHeader])
        return manhattanDistance

    #Computes the Euclidean distance between two vectors in dictionary form with numeric values in it.
    #Return the Euclidean distance between two vectors
    def euclidean(self, vector1, vector2):
        euclideanDistance = 0
        distanceSquare = 0
        for colHeader in self.colHeader:
            if (type(vector1[colHeader])  is not str) and (vector1[colHeader]  is not None):
                if (type(vector2[colHeader])  is not str) and (vector2[colHeader]  is not None):
                    distanceSquare += (vector1[colHeader] - vector2[colHeader])**2
        euclideanDistance = sqrt(distanceSquare)
        return euclideanDistance

    #Computes the Pearson Coeff between two vectors in dictionary form with numeric values in it.
    #Function not used. Can be used instead of Manhattand Distance / Euclidean Distance
    def pearson(self, rating1, rating2):
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        n = 0
        for key in rating1:
            if key in rating2:
                n += 1
                x = rating1[key]
                y = rating2[key]
                sum_xy += x * y
                sum_x += x
                sum_y += y
                sum_x2 += pow(x, 2)
                sum_y2 += pow(y, 2)
        # now compute denominator
        denominator = sqrt(sum_x2 - pow(sum_x, 2) / n) * sqrt(sum_y2 - pow(sum_y, 2) / n)
        if denominator == 0:
            return 0
        else:
            return (sum_xy - (sum_x * sum_y) / n) / denominator
        
    def nearestNeighbor(self, tickerValList, tickerDict):
        """creates a sorted list of 10 nearest neigbors to tickerVal from tickerDict"""
        """returns a dictionary of the form {'AAPL':{'AGN':distance,'CSCO':distance}}"""
        distanceList = {}
        for tickerVal in tickerValList:
            vector1 = tickerDict[tickerVal]
            distancearrayList = {}
            for tickerVal2 in tickerDict:
                vector2 = tickerDict[tickerVal2]
                distance = self.euclidean(vector1,vector2)
                distancearrayList[tickerVal2] = distance
            distanceList[tickerVal] = distancearrayList                   
        return distanceList
        
    #Take a list of numbers/floats and return the median
    def getMedian(self, alist):
        """Return median of a list of Integers"""
        length = 0
        median = 0.0
        #lenght of the list
        length = len(alist)
        #sort the list
        alist.sort()
        if (length % 2) == 0:
            median = float(alist[abs(length/2)] + alist[abs(length/2) - 1])/2            
        else:
            median = float(alist[abs(length/2)])
        return median
        
    #Take a list of numbers/floats, median and return the absolute standard deviation
    def getAbsoluteStandardDeviation(self, alist, median):
        """given alist and median return absolute standard deviation"""
        sum = 0
        asd = 0
        for i in range(len(alist)):
            sum += abs(alist[i] - median)
        asd = sum / len(alist)
        return asd 

    #Take a list of string objects and return the same list stripped of extra whitespace.
    def stripList(self, alist):
        return([x.strip() for x in alist])
        
    #Check if the number arguament passed is a number. Returns True/False
    def isNumber(self, number):
        try:
            float(number)
            return True
        except ValueError:
            return False
            
    #Takes ticker and distance dictionary as inputs. 
    #Returns an sorted list of K neighbors: nearest or farthest. 
    #Also returns a neighbor dictionary of form '{Ticker':Distance}. For e.g.
    """{'TWC': 8.038591581629646, 'VNTV': 7.411760125067703, 'ADS': 6.50817121303119, 
        'VIAB': 7.163646547168478, 'K': 8.505894614773853, 'DE': 8.251429145250839, 
        'AXP': 7.939314955225916, 'KRFT': 8.435903431260883, 'WU': 8.47180658183811, 
        'VZ': 0.0, 'DTV': 7.907489692933219}"""
    #By default sort by nearest and K = 5
    def sortNeighborList(self, ticker, distanceDict, k = 5, descFlag = False):
        neighborList = []
        neighborDict = {}
        if descFlag == True:
            #sortedList = sorted([(value,key) for (key,value) in distanceDict[ticker].items()], reverse = True)
            sortedList = sorted(distanceDict[ticker].items(), key=lambda x: x[1], reverse = True)
        else:
            sortedList = sorted(distanceDict[ticker].items(), key=lambda x: x[1])
        print("\nTicker: " + ticker)
        print("(" + str(k) + ")Nearest Neighbors \tEuclidean Distance")
        for i in range(k+1):
            neighborList.append(sortedList[i])
            neighborDict[sortedList[i][0]] = sortedList[i][1]
            print("\t" + str(sortedList[i][0]) + "\t\t\t" + str(sortedList[i][1]))
        return neighborList, neighborDict
    
    #Takes the attribute to be predicted, along with list and dictionary of nearest neighbors
    #Uses the global nonNormalizedTickerDict variable to iterate through the attributes for each neighbor
    #since we need non Normalized values of the attributes in question
    def predictAttribute(self, attribute, neighborList, neighborDict):
        nonNormalizedTickerDict = self.nonNormalizedTickerDict
        inverDistDict = {}
        influenceDict = {}
        for ticker in neighborDict:
            #Finding inverse of everyelement in the neighborDict, except for the one with 0.
            if neighborDict[ticker] != 0:
                inverDist = 1/neighborDict[ticker]
                inverDistDict[ticker] = inverDist
            else:
                predictedTicker = ticker
        sumInverDist = sum(list(inverDistDict.values()))

        print("\nInfluence Summary for Predictive Attribute: "+ str(attribute))
        print("Tickers \t"+ str(attribute) +"\tInfluence")
        predictedValue = 0.0
        for ticker in inverDistDict:
            if(inverDistDict[ticker] != 0):
                influenceDict[ticker] = inverDistDict[ticker]/sumInverDist
                if nonNormalizedTickerDict[ticker][attribute] == '':
                    attributeValue = 0.0
                    influencePercent = 0.0
                    predictedValue += (influenceDict[ticker] * attributeValue)
                    attributeValue = None
                else:                    
                    attributeValue = float(nonNormalizedTickerDict[ticker][attribute])
                    influencePercent = round(influenceDict[ticker] * 100,2)
                    predictedValue += (influenceDict[ticker] * attributeValue)
                print(ticker + "\t\t" + str(attributeValue) +"\t"+ str(influencePercent)+"%")                    
        print("\n" + predictedTicker + " predicted " + str(attribute) + " is = %0.2f" % predictedValue)
        print(predictedTicker + " current market " + str(attribute) + " is = %0.2f" % float(self.nonNormalizedTickerDict[predictedTicker][attribute]))
        return (predictedTicker, predictedValue)

        
#Unit Test of Spyder Class
startTime = datetime.now()
print(startTime.strftime('%H:%M:%S %m/%d/%Y') + " :: Starting Predictor")
#Replace with file path of SP500 CSV file
filepath = "/Users/Viju/Dropbox/PythonWorkSpace/SP500_Analysis/"
#For SP500 CSV file
filename = "SP500_Data.csv"
filename = filepath + filename

predObj = PePredictor(filename)
tickerDict = predObj.loadTickerDict()
print("No. of Tickers loaded to Ticker Dictionary: %d" % predObj.lineCount)

tickerDict = predObj.formatColumn(predObj.colHeader,tickerDict)
tickerDict = predObj.normalizeColumn(predObj.colHeader, tickerDict)
distDict = predObj.nearestNeighbor(predObj.rowHeader, tickerDict)

nearestNeighborCnt = 10 # Default set to 5
ticker = 'AAPL'
ticker = ticker.upper()
#Attribute should belong to this attributeList = ['P/E','P/B','P/S']
attribute = 'P/E'
(neighborList, neighborDict) = predObj.sortNeighborList(ticker, distDict, nearestNeighborCnt)
predObj.predictAttribute(attribute, neighborList, neighborDict)

endTime = datetime.now()
runTime = endTime - startTime
print("\n")
print(endTime.strftime('%H:%M:%S %m/%d/%Y') + " :: Ending Predictor")
print("Predictor Run Time " + str(runTime)[:7])