import sys
import sparkconnect


try:

    spContext = sparkconnect.getContext()

    spSession = sparkconnect.getSession()

    autoData = spContext.textFile('data/auto-miles-per-gallon.csv')

    #print(autoData.take(5))


except ImportError as ie:
    print(ie)

except Exception as e:
    print(e)


# Remove header from the imported flat files
def excludeHeader(autoRDD):

    header = autoRDD.first()

    autoRows = autoRDD.filter(lambda x:x!=header)

    return autoRows


