# -*- coding: utf-8 -*-
"""
Created on Tue Aug 5 18:45:52 2014

@author: Viju
"""
import numpy as np
import pandas as pd
import pandas.io.data as web
from math import sqrt
from datetime import datetime

#Define Global constants
CONST_CLFUND= 'Closed-End Fund - Equity'
CONST_ETF = 'Exchange Traded Fund'
CONST_COUNTRY = 'USA'
CONST_MARKET_CAP = 5000                         #Market Cap = 5000 => SPY750
CONST_EPS = 'EPS (ttm)'
CONST_MON_VOLATILITY = 'Volatility (Month)'
CONST_DROP_COL = ['Market Cap','Current Ratio','Quick Ratio','LT Debt/Equity','Total Debt/Equity',
                  'Shares Outstanding','Shares Float','Insider Ownership','Insider Transactions',
                  'Institutional Ownership','Institutional Transactions','Float Short','Short Ratio',
                  'Performance (Week)','Average True Range','Volatility (Week)','20-Day Simple Moving Average',
                  '50-Day High','50-Day Low','52-Week High','52-Week Low','Change from Open','Gap',
                  'Analyst Recom','Average Volume','Relative Volume','Change','Volume','Earnings Date','Price']
CONST_DROP_ROW = ['GOOG']               #Preferntial Shares in SPY. Immaterial to Correlations
                  
class PePredictor:
    #Constructor
    def __init__(self, filename):
        self.lineCount = 0
        #normalized global object
        self.nonNormalizedTickerDF = {} 
        fileDataframe = self.loadDataframe(filename)
        #Get rowTickers headers from rowvalues in the dataframe
        #Get colHeaders from the columnValues of the dataframe
        self.colHeaders = list(fileDataframe.columns.values)
        self.rowTickers = list(fileDataframe.index.values)
        print("Total Loaded No# of Tickers: " + str(len(self.rowTickers)))
        print("Total Loaded No# of Attributes: " + str(len(self.colHeaders)))
        print("Attributes used for Correlation:" + str(self.colHeaders[4:]))
        #print self.rowTickers
        self.nonNormalizedTickerDF = fileDataframe            
        
    #loadDataFrame from given CSV File. Currently loads from CSV placed in the same directory.
    #Filters based on the Market Cap, Country Name. No Close Equities and ETF
    def loadDataframe(self, filename = ''):
        if(filename == ''):
            print("File not found. Check if the path of the file is valid")
            return        
        fileDataframe = pd.read_csv(filename)
        #Replace the row indexes with Ticker names as row names
        fileDataframe.index = fileDataframe['Ticker']
        fileDataframe.drop('Ticker', axis=1, inplace=True)
        #Filerting by Market Cap and Country        
        fileDataframe = fileDataframe[(fileDataframe['Country'] == CONST_COUNTRY) & 
                                      (fileDataframe['Market Cap'] >= CONST_MARKET_CAP)]
        fileDataframe = fileDataframe[fileDataframe['Industry'] != CONST_ETF] 
        fileDataframe = fileDataframe[fileDataframe['Industry'] != CONST_CLFUND] 
        #Dropping list of CONST_DROP_COL/ROW columns & rows to skip them in the Correlations
        fileDataframe.drop(CONST_DROP_ROW, axis=0, inplace=True)
        fileDataframe.drop(CONST_DROP_COL, axis=1, inplace=True)
        #print fileDataframe.head()
        return fileDataframe

    #Function to Normalize the non String columns and return the normalized Dataframe object    
    def normalizeDataframe(self, dframeObj):
        df = dframeObj
        """given a column name, normalize that column in Dataframe
        using the Modified Standard Score"""
        for colHeader in self.colHeaders:
           #extract values of column Header to list and ignore object data type columns
            col = df[colHeader]
            if col.dtype != object:
                median = col.median()
                #drop the NaN from the column before packaging it into the asd column list
                colValues = col.dropna().values                    
                asd = self.getAbsoluteStandardDeviation(colValues,median)
                col = (col - median)/asd
                df[colHeader] = col
            elif (col.dtype == object) and ('%' in col[0]):
                col = col.str[0:-1]
                col = col.astype(float)
                median = col.median()
                #drop the NaN from the column before packaging it into the asd column list
                colValues = col.dropna().values                    
                asd = self.getAbsoluteStandardDeviation(colValues,median)
                col = (col - median)/asd
                df[colHeader] = col
        return df

    #Function to return absolute standard deviation from a list of numerica values and their median
    def getAbsoluteStandardDeviation(self, alist, median):
        """given alist and median return absolute standard deviation"""
        sum = 0
        asd = 0
        for i in range(len(alist)):
            sum += abs(alist[i] - median)
        asd = sum / len(alist)
        return asd 
        
    #Function to returns the correlation matrix as the dataframe
    def calcPearsonCorrelation(self, dframeObj):
        transposeDf = dframeObj.transpose()
        #Filtering from Column 4 onvwards which consists of the numeric columns
        transposeDf = transposeDf[4:]
        transposeDf = transposeDf.astype(float)
        corrMatrixDf = transposeDf.corr()
        return corrMatrixDf
 
    #Function takes Ticker, Correlation Matrix, K neighbors as an argument
    #Returns a series of K nearest neighbor for given ticker. Defaults to 5 nearest neighbors
    def nearestNeighbor(self, ticker, corrMatrix, k=5):
        print("\n"+str(k) + " Nearest Neighbors for: " + ticker)
        k += 1
        ticker = ticker.upper()
        nearestNeigborSr = corrMatrix[ticker]
        nearestNeigborSr = nearestNeigborSr.sort(ascending = False, inplace = False)[0:k]       
        print nearestNeigborSr
        return nearestNeigborSr
      
    #Function takes predicter Attribute, Nearest Neighbor Series and non normalized fileDataFrame
    #Returns predicted Ticker and predicted Attribute
    def predictAttribute(self, ticker, attribute, nearestNeigborSr, nonNormalizedDataframe):
        predictedAttr = 0
        assert(ticker == str(nearestNeigborSr.index[0]))
        neighborInfluence = nearestNeigborSr[1:]/sum(nearestNeigborSr[1:]) * 100
        assert(round(sum(neighborInfluence),2) == 100.0)        
        
        print("\nInfluence Summary for Predictive Attribute: "+ str(attribute))
        print("Tickers \t"+ str(attribute) +"\tInfluence")
        
        for index, influence in neighborInfluence.iteritems():                      
            attributeVal = nonNormalizedDataframe.ix[index][attribute]
            if not np.isnan(attributeVal):
                predictedAttr += influence/100 * attributeVal
            print(str(index) + "\t\t" + str(attributeVal) +"\t"+ str(round(influence,2))+"%")
            
        print(ticker + " Predicted " + attribute + " : %0.2f " % predictedAttr)
        return predictedAttr

    #Takes the ticker and the Predicted PE for the relevant ticker to generate the Price Confidence.
    #User yahoo finance to get the webquote of a Ticker
    def generatePriceConfidence(self, ticker, predictedPE, fileDf):        
        print("\nPrice Confidence Summary for Ticker: " + ticker)
        mrktData = web.get_quote_yahoo(ticker)
        mrktData.columns = ['P/E', 'Change Percent', 'Current Price', 'Short Ratio', 'Time']
        mrktPrice = mrktData.ix[ticker]['Current Price']
        print mrktData

        upperPredPrice = 0.0
        lowerPredPrice = 0.0
        mrktEPS = float(fileDf.ix[ticker][CONST_EPS])
        mrktMonVolatility = float(fileDf.ix[ticker][CONST_MON_VOLATILITY][:-1])
        mrktYrVolatility = mrktMonVolatility * sqrt(12)
        predictedPrice = predictedPE * mrktEPS
        print("\nEPS (ttm) = %0.2f" % mrktEPS)
        print("Volatility (Month) = %0.2f" % mrktMonVolatility + "%")
        print("Volatility (Year) = %0.2f" % mrktYrVolatility + "%")
        print("Market Price = %0.2f" % mrktPrice)
        print("Predicted Price (Mean) = %0.2f" % predictedPrice)
        confIntVolatility = 1.96 * mrktYrVolatility
        upperPredPrice = predictedPrice * (1 + confIntVolatility/100)
        lowerPredPrice = predictedPrice * (1 - confIntVolatility/100)
        print("\tConf Int \t95% or 1.96σ")
        print("Volatility (Year) \t %0.2f" % confIntVolatility + "%")
        print("Upper Price \t\t %0.2f" % upperPredPrice)
        print("Lower Price \t\t %0.2f" % lowerPredPrice)
        print("\nValuation Summary for Ticker: " + ticker)
        if mrktPrice >= upperPredPrice:
            print("\tOver Valued")
            valPercentage = ((mrktPrice - upperPredPrice)/mrktPrice) * 100
            print("\tDownside of %0.2f" % valPercentage + "% to Upper Predicted Price")
            valPercentage = ((mrktPrice - predictedPrice)/mrktPrice) * 100
            print("\tDownside of %0.2f" % valPercentage + "% to Mean Predicted Price")
        elif mrktPrice < lowerPredPrice:
            print("\tUnder Valued")
            valPercentage = ((lowerPredPrice - mrktPrice)/mrktPrice) * 100
            print("\tUpside of %0.2f" % valPercentage + "% to Lower Predicted Price")
            valPercentage = ((predictedPrice - mrktPrice)/mrktPrice) * 100
            print("\tUpside of %0.2f" % valPercentage + "% to Mean Predicted Price")
        elif mrktPrice >= predictedPrice:
            print("\tFairly Valued")
            valPercentage = ((mrktPrice - predictedPrice)/mrktPrice) * 100
            print("\tDownside of %0.2f" % valPercentage + "% to Mean Predicted Price")
        elif mrktPrice < predictedPrice:
            print("\tFairly Valued")
            valPercentage = ((predictedPrice - mrktPrice)/mrktPrice) * 100
            print("\tUpside of %0.2f" % valPercentage + "% to Mean Predicted Price")
        else:
            print("Oops ! Something went wrong. Cannot value Ticker : " +ticker)   

#Package Function to log Start Time        
def startTimeLog():
    startTime = datetime.now()
    print(startTime.strftime('%H:%M:%S %m/%d/%Y') + " :: Starting Predictor")
    return startTime

#Package Function to log End Time and Run Time    
def endTimeLog(startTime = ''):
    endTime = datetime.now()
    print("\n")
    print(endTime.strftime('%H:%M:%S %m/%d/%Y') + " :: Ending Predictor")
    if startTime != '' :       
        runTime = endTime - startTime
        print("Predictor Run Time " + str(runTime)[:7])
        
#Unit Test of PEPredictor Class
startTime = startTimeLog()
filepath = "/Users/Viju/Dropbox/PythonWorkSpace/SP500_Analysis/PEPredictor/"
filename = "SPYALL_Data.csv"
filename = filepath + filename

predObj = PePredictor(filename)
fileDf = predObj.nonNormalizedTickerDF
predObj.normalizeDataframe(fileDf)
corrMatrix = predObj.calcPearsonCorrelation(fileDf)

kNeighbors = 10
ticker = 'GILD'
ticker = ticker.upper()
#Attribute should belong to this attributeList = ['P/E','P/B','P/S','Dividend Yield']
attribute = 'P/E'
attribute = attribute.upper()
nearestNeigborSr = predObj.nearestNeighbor(ticker, corrMatrix, kNeighbors)
#Load the original non-Normalized Dataframe again from file
fileDf = predObj.loadDataframe(filename)
predictedPE = predObj.predictAttribute(ticker, attribute, nearestNeigborSr, fileDf)
#Price confidence can only be generate when attribute is P/E
predObj.generatePriceConfidence(ticker, predictedPE, fileDf)

endTimeLog(startTime)