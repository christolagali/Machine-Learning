import sys
import sparkconnect
from pyspark.sql import Row
import matplotlib.pyplot as plt
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression

from pyspark.ml.evaluation import RegressionEvaluator

# Remove header from the imported flat files
def excludeHeader(autoRDD):

    header = autoRDD.first()

    autoRows = autoRDD.filter(lambda x:x!=header)

    return autoRows

def cleanupData(autoStr):
    
    global avgHP
    
    # find ? and replace with average value
     
    autoLst = autoStr.split(",")
    
    if autoLst[3] == "?":
        autoLst[3] = avgHP.value
        
    # create a row with converted float values
    values = Row(MPG=float(autoLst[0]), CYLINDERS=float(autoLst[1]), DISPLACEMENT=float(autoLst[2]), HORSEPOWER=float(autoLst[3]), WEIGHT=float(autoLst[4]), ACCELERATION=float(autoLst[5]), MODELYEAR=float(autoLst[6]), NAME=autoLst[7])

    return values

def transformToLabeledPoint(row):

    lp = (row['MPG'],Vectors.dense([row['ACCELERATION'],row['DISPLACEMENT'],row['WEIGHT']]))
    
    return lp



##########################################################################################################################
# Main Program
##########################################################################################################################
try:

     
    # get spark context

    sc = sparkconnect.getContext()
    sp = sparkconnect.getSession()

    
    # loading and prepping the data

    autoRDD = sc.textFile('../data/auto-miles-per-gallon.csv')

    header = autoRDD.first()

    autoDataRDD = autoRDD.filter(lambda x : x != header)


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

    autoMap = autoDataRDD.map(cleanupData)

    autoMap.persist()

    # create a Data Frame

    autoDF = sp.createDataFrame(autoMap)



    ###############################################################################################################
    # Perform Data Analysis
    #
    ###############################################################################################################

    # find correlation coefficient

    """for i in autoDF.columns:
        if not isinstance(autoDF.select(i).take(1)[0][0],str):
            print('Correlation of MPG with ',i,autoDF.stat.corr('MPG',i))"""
    

    # Converting Data Frame to Pandas Data Frame
    autoPandasDF = autoDF.toPandas()

    
    # Plotting MPG (target variable) against other independent Variables for better Corelation

    # MPG & ACCELERATION
    autoPandasDF.plot(kind='scatter',x='ACCELERATION',y='MPG')
    plt.scatter(autoPandasDF.ACCELERATION,autoPandasDF.MPG)
    plt.xlabel('ACCELERATION')
    plt.ylabel('MPG')
    plt.savefig('charts/MPG_ACCELERATION.png')

    # MPG & DISPLACEMENT
    autoPandasDF.plot(kind='scatter',x='DISPLACEMENT',y='MPG')
    plt.scatter(autoPandasDF.DISPLACEMENT,autoPandasDF.MPG)
    plt.xlabel('DISPLACEMENT')
    plt.ylabel('MPG')
    plt.savefig('charts/MPG_DISPLACEMENT.png')

    # MPG & WEIGHT
    autoPandasDF.plot(kind='scatter',x='WEIGHT',y='MPG')
    plt.scatter(autoPandasDF.WEIGHT,autoPandasDF.MPG)
    plt.xlabel('WEIGHT')
    plt.ylabel('MPG')
    plt.savefig('charts/MPG_WEIGHT.png')

    # MPG & CYLINDERS
    autoPandasDF.plot(kind='scatter',x='CYLINDERS',y='MPG')
    plt.scatter(autoPandasDF.CYLINDERS,autoPandasDF.MPG)
    plt.xlabel('CYLINDERS')
    plt.ylabel('MPG')
    plt.savefig('charts/MPG_CYLINDERS.png')

    # MPG & HORSEPOWER
    autoPandasDF.plot(kind='scatter',x='HORSEPOWER',y='MPG')
    plt.scatter(autoPandasDF.HORSEPOWER,autoPandasDF.MPG)
    plt.xlabel('HORSEPOWER')
    plt.ylabel('MPG')
    plt.savefig('charts/MPG_HORSEPOWER.png')


    #####################################################################################################################
    # Prepare Data for ML
    ##################################################################################################################

    


    autoLabeledPoint = autoMap.map(transformToLabeledPoint)

    autoLabeledPoint.persist()

    #print(autoLabeledPoint.take(5))


    # IMP -	All Machine Learning Algorithms in Spark expect our data to be in a Data Frame that contains the label and features columns.

    autoMLDF = sp.createDataFrame(autoLabeledPoint,['label','features'])

    #autoMLDF.show(5)


    #......................................................................................................
    # actual Machine learniing
    #......................................................................................................

    # split the dataset into Training and Testing Data set

    # splits data randomly. % of split. 90% goes to training and 10% goes to testing
    (trainData,testData) = autoMLDF.randomSplit([0.6,0.4])

    # train count
    #print(trainData.count())

    # test count 
    #print(testData.count())


    #######################################
    ##  Train Model
    ######################################

    # maxIter = no of times it is going to Iterate to build the model
    # more iterations would mean a better but more time too
    lin_reg = LinearRegression(maxIter=30)

    # fit() to build the model

    line_reg_model = lin_reg.fit(trainData)

    # Model will basically contain Coefficient (alpha) and Intercept(Beta)
    print("Coefficient : " ,str(line_reg_model.coefficients))
    print("Intercept is :" ,str(line_reg_model.intercept))


    predictions  = line_reg_model.transform(testData)

    predictions.persist()
    #print(predictions.collect())

    # this means our equation will look as follows:
    # MPG(target variable) =  (0.18070324555844722 * ACCELERATION)  (-0.015865508002443303 * DISPLACEMENT) (-0.005853837529094065 * WEIGHT) + 41.1518935140389 (INTERCEPT)

    # Once we build the model we can test the model

    predictions.select("prediction","label","features").show()



    #####################################################
    # Evaluate Results of the predictions
    #####################################################
    
    # Metric Name R2 0.6976322649426134
    evaluator = RegressionEvaluator(predictionCol='prediction',labelCol='label',metricName="r2")

    print(evaluator.evaluate(predictions))



except ImportError as ie:
    print(ie)

except Exception as e:
    print(e)



