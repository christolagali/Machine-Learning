import sparkconnect
import sys
from pyspark.sql import Row
import matplotlib.pyplot as plt
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Exclude header from RDD
def excludeHeader(bankRDD):

    header = bankRDD.first()

    bankRows = bankRDD.filter(lambda x: x!=header)

    return bankRows


def cleanData(bankStr):

    bankLst = bankStr.split(",")

    age = float(bankLst[0])

    if bankLst[2] == '"married"':
        married = 1.0
    elif bankLst[2] == '"single"':
        married = 2.0
    else:
        married = 3.0
    
    if bankLst[3] == '"primary"':
        education = 1.0
    elif bankLst[3] == '"secondary"':
        education = 2.0
    else:
        education = 3.0
    

    default = 0.0 if bankLst[4] == '"no"' else 1.0

    balance = float(bankLst[5])

    loan = 0.0 if bankLst[7] == '"no"' else 1.0

    outcome = 0.0 if bankLst[16] == '"no"' else 1.0

    return Row(AGE = age,MARRIED = married,EDUCATION=education,DEFAULT=default,BALANCE=balance,LOAN=loan,OUTCOME=outcome)


def transformForML(row):

    featureArray = []

    featureArray.append(row['AGE'])
    featureArray.append(row['BALANCE'])
    featureArray.append(row['DEFAULT'])
    featureArray.append(row['EDUCATION'])
    featureArray.append(row['LOAN'])
    featureArray.append(row['MARRIED'])

    return (row['OUTCOME'],Vectors.dense(featureArray))




try:

    sc = sparkconnect.getContext()
    sp = sparkconnect.getSession()


    bankData = sc.textFile('../data/bank_data.csv')

    #print(bankData.take(5))

    bankData.persist()

    bankRDD = excludeHeader(bankData)

    #print(bankRDD.take(5))


    # clean data and encapsulate into  Row()

    bankCleanRDD = bankRDD.map(cleanData)

    bankCleanRDD.persist()



    ###############################################################################################################
    # Perform Data Analysis
    #
    ###############################################################################################################

    # convert to DataFrame and Pandas Data Frame

    bankDF = sp.createDataFrame(bankCleanRDD)

    bankPandasDF = bankDF.toPandas()

    #bankDF.show(10)

    ###### Get stats
    #bankPandasDF.describe().show()


    #########################################################
    #####
    #####   Scatter plots for each variable against OUTCOME
    #########################################################

    
    fig,ax = plt.subplots()
    ax.scatter(x=bankPandasDF.AGE,y=bankPandasDF.OUTCOME,c=bankPandasDF.OUTCOME)
    ax.xlabel('AGE')
    ax.ylabel('OUTCOME')
    ax.legend()
    plt.savefig('charts/OUTCOME_AGE.png')



    plt.scatter(x=bankPandasDF.BALANCE,y=bankPandasDF.OUTCOME,c=bankPandasDF.OUTCOME)
    plt.xlabel('BALANCE')
    plt.ylabel('OUTCOME')
    plt.legend()
    plt.savefig('charts/OUTCOME_BALANCE.png')



    plt.scatter(x=bankPandasDF.DEFAULT,y=bankPandasDF.OUTCOME,c=bankPandasDF.OUTCOME)
    plt.xlabel('DEFAULT')
    plt.ylabel('OUTCOME')
    plt.legend()
    plt.savefig('charts/OUTCOME_DEFAULT.png')


    plt.scatter(x=bankPandasDF.EDUCATION,y=bankPandasDF.OUTCOME,c=bankPandasDF.OUTCOME)
    plt.xlabel('EDUCATION')
    plt.ylabel('OUTCOME')
    plt.savefig('charts/OUTCOME_EDUCATION.png')



    plt.scatter(x=bankPandasDF.LOAN,y=bankPandasDF.OUTCOME,c=bankPandasDF.OUTCOME)
    plt.xlabel('LOAN')
    plt.ylabel('OUTCOME')
    plt.legend()
    plt.savefig('charts/OUTCOME_LOAN.png')



    plt.scatter(x=bankPandasDF.MARRIED,y=bankPandasDF.OUTCOME,c=bankPandasDF.OUTCOME)
    plt.xlabel('MARRIED')
    plt.ylabel('OUTCOME')
    plt.legend()
    plt.savefig('charts/OUTCOME_MARRIED.png')

    del plt
    





    #########################################################
    #####
    #####   Correlation Coefficient for each variable
    #########################################################
    """
    for i in bankDF.columns:
        if not isinstance(bankDF.select(i).take(1)[0][0],str):
            print("Correlation with Outcome is ",i,bankDF.stat.corr('OUTCOME',i))
    """


    #########################################################
    #####
    #####   Perform Machine Learning
    #########################################################

    # tranforming RDD to Labeled Point
    bankML = bankCleanRDD.map(transformForML)

    bankML.persist()


    # Converting to DataFrame

    bankMLDF = sp.createDataFrame(bankML,['label','features'])

    #print(bankML.take(5))

    # Splitting data into Training and test datasets
    (train,test) = bankMLDF.randomSplit([0.7,0.3])


    # train data counts # 375
    # print(train.count())

    # test data count   # 166
    # print(test.count())


    logreg = LogisticRegression(labelCol='label',featuresCol='features',maxIter=10)


    # create a model
    log_reg_model = logreg.fit(train)

    predictions = log_reg_model.transform(test)

    predictions.persist()


    #.................................................................................
    #   Coefficients  DenseMatrix([
    # [-3.38742717e-02,  3.04807506e-05, -8.07349111e-01, 2.15245871e-01,  2.22353603e-01,  5.83806692e-01]])
    #   Intercepts [-0.46070628895829485]
    #   Area Under ROC  0.6674415204678364
    #   
    #.................................................................................
    print('Coefficients ',log_reg_model.coefficientMatrix)
    print('Intercepts' , log_reg_model.interceptVector)

    
    # Area Under ROC
    trainingSummary = log_reg_model.summary

    roc = trainingSummary.roc.toPandas()
    plt.plot(roc['FPR'],roc['TPR'])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()


    print('Area Under ROC ', trainingSummary.areaUnderROC )
    


    bin_evaluator = BinaryClassificationEvaluator()

    # Test Area Under ROC  0.6444015444015446

    print('Test Area Under ROC ',bin_evaluator.evaluate(predictions))




except ImportError as ie:
    print(ie)

except Exception as e:
    print(e)