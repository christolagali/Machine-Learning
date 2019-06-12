# -*- coding: utf-8 -*-
"""
Make sure you give execute privileges
-----------------------------------------------------------------------------

           Spark with Python: Setup Spyder IDE for Spark

             Copyright : V2 Maestros @2016
                    
Execute this script once when Spyder is started on Windows
-----------------------------------------------------------------------------
"""


import os
import sys

# NOTE: Please change the folder paths to your current setup.
#Windows
if sys.platform.startswith('win'):
    #Where you downloaded the resource bundle
    os.chdir("C:/SparkML/sparkML_Pracs/linear_Regression")
    #Where you installed spark.    
    os.environ['SPARK_HOME'] = 'C:/spark-2.3.1-bin-hadoop2.6'
#other platforms - linux/mac
else:
    os.chdir("/Users/kponnambalam/Dropbox/V2Maestros/Modules/Apache Spark/Python")
    os.environ['SPARK_HOME'] = '/users/kponnambalam/products/spark-2.0.0-bin-hadoop2.7'

os.curdir



# Create a variable for our root path
SPARK_HOME = os.environ['SPARK_HOME']

#Add the following paths to the system path. Please check your installation
#to make sure that these zip files actually exist. The names might change
#as versions change.
sys.path.insert(0,os.path.join(SPARK_HOME,"python"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib","pyspark.zip"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib","py4j-0.10.7-src.zip"))

#Initialize SparkSession and SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkContext




#Create a Spark Session
SpSession = SparkSession \
    .builder \
    .master("local[2]") \
    .appName("Chris App") \
    .config("spark.executor.memory", "1g") \
    .config("spark.cores.max","2") \
    .getOrCreate()
    
#Get the Spark Context from Spark Session    
SpContext = SpSession.sparkContext

def getContext():
    SpContext = SpSession.sparkContext
    return SpContext

def getSession():
    return SpSession
