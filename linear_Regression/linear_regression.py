import sys
import sparkconnect
from pyspark.sql import Row

# Remove header from the imported flat files
def excludeHeader(autoRDD):

    header = autoRDD.first()

    autoRows = autoRDD.filter(lambda x:x!=header)

    return autoRows


# clean Data Method
def cleanData(autoStr):

    global avgHP
    autoLst = autoStr.split(',')

    if autoStr[3] == '?':
        hpValue = avgHP
    else:
        hpValue = autoStr[3]
    
    return Row(MPG=float(autoLst[0]),CYLINDERS=float(autoLst[1]),DISPLACEMENT=float(autoLst[2]),HORSEPOWER=float(hpValue),WEIGHT=float(autoLst[4]),ACCELERATION=float(autoLst[5]),MODELYEAR=float(autoLst[6]),NAME=autoLst[7])




##########################################################################################################################
# Main Program
##########################################################################################################################
try:

    spContext = sparkconnect.getContext()

    spSession = sparkconnect.getSession()

    autoData = spContext.textFile('data/auto-miles-per-gallon.csv')

    #print(autoData.take(5))

    # Broadcast variable
    avgHP = spContext.broadcast(80)

    # cleaned RDD
    autoRDD = excludeHeader(autoData)

    autoCleanRDD = autoRDD.map(cleanData)

    print(autoCleanRDD.take(5))




except ImportError as ie:
    print(ie)

except Exception as e:
    print(e)



