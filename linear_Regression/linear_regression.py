import sys
import sparkconnect
from pyspark.sql import Row
import matplotlib.pyplot as plt
from pyspark.ml.linalg import Vectors

# Remove header from the imported flat files
def excludeHeader(autoRDD):

    header = autoRDD.first()

    autoRows = autoRDD.filter(lambda x:x!=header)

    return autoRows







##########################################################################################################################
# Main Program
##########################################################################################################################
try:

     
    # get spark context

    sc = sparkconnect.getContext()
    sp = sparkconnect.getSession()

    
    # loading and prepping the data

    autoRDD = sc.textFile('data/auto-miles-per-gallon.csv')

    header = autoRDD.first()

    autoDataRDD = autoRDD.filter(lambda x : x != header)
    #print(autoRDD.take(5))

    #print(autoDataRDD.take(5))

    #.........................................................................................................
    #   ['MPG,CYLINDERS,DISPLACEMENT,HORSEPOWER,WEIGHT,ACCELERATION,MODELYEAR,NAME', 
    #   '18,8,307,130,3504,12,70,chevrolet chevelle malibu'
    #   
    #   we wil try to predict MPG (target) based using features (rest of the attributes)
    #.........................................................................................................


    # Data Cleanup

    # handle missing data (like HP column)
    # converting nos from string to float

    # creating a broadcast variable for the average HP
    avgHP = sc.broadcast(80.0)

    def cleanupData(autoStr):

        global avgHP
        # find ? and replace with average value

        autoLst = autoStr.split(",")

        if autoLst[3] == "?":
            autoLst[3] = avgHP.value
        
        # create a row with converted float values
        values = Row(MPG=float(autoLst[0]), CYLINDERS=float(autoLst[1]), DISPLACEMENT=float(autoLst[2]), HORSEPOWER=float(autoLst[3]), WEIGHT=float(autoLst[4]), ACCELERATION=float(autoLst[5]), MODELYEAR=float(autoLst[6]), NAME=autoLst[7])

        return values

    autoMap = autoDataRDD.map(cleanupData)

    autoMap.persist()

    #print(autoMap.take(10))

    # create a Data Frame

    autoDF = sp.createDataFrame(autoMap)

    #autoDF.show()


    ###############################################################################################################
    # Perform Data Analysis
    #
    ###############################################################################################################

    # find correlation coefficient

    """for i in autoDF.columns:
        if not isinstance(autoDF.select(i).take(1)[0][0],str):
            print('Correlation of MPG with ',i,autoDF.stat.corr('MPG',i))"""
    

    autoPandasDF = autoDF.toPandas()

    #autoPandasDF.plot(kind='scatter',x='DISPLACEMENT',y='MPG')


    #plt.scatter(autoPandasDF.DISPLACEMENT,autoPandasDF.MPG)

    #plt.show()


    #####################################################################################################################
    # Prepare Data for ML
    ##################################################################################################################

    def transformToLabeledPoint(row):

        lp = (row['MPG'],Vectors.dense([row['ACCELERATION'],row['DISPLACEMENT'],row['WEIGHT']]))
    
        return lp


    autoLabeledPoint = autoMap.map(transformToLabeledPoint)

    print(autoLabeledPoint.take(5))






except ImportError as ie:
    print(ie)

except Exception as e:
    print(e)



