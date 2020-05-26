import sys
import math
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree


def SetPath(sc):
    global Path
    if sc.master[0:5] == "local":
        Path = "file:/usr/local/py_case/pj_UCI_BikeSharing/"
    else:
        Path = "hdfs://master:9000/py_case/PythonProject_3/"


def SetLogger(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)


def CreateSparkContext():
    sparkConf = SparkConf() \
        .setAppName("RecommendTrain") \
        .set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf=sparkConf)
    SetPath(sc)
    SetLogger(sc)
    print("master=" + sc.master)
    return sc

def convert_float(field):
    return (0 if field=="?" else float(field))


def extract_label(record):
    label = (record[-1])
    return float(label)

def extract_features(record, featureEnd):
    seasonFeature = [convert_float(field) for field in record[2]]
    features = [convert_float(field) for field in record[4: featureEnd - 2]]
    return np.concatenate((seasonFeature,features))

def PrepareData(sc):
    print("=========loading and transfer data===============")
    print("=========build RDD for training and accessing============")
    print("=========seperate three parts as trainData, validationData and train data=======")
    print("=========now begin to load the data=======")
    rawDataWithHeader = sc.textFile(Path + "data/hour.csv")
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    print("total " + str(rawData.count()))
    lines = rawData.map(lambda x: x.split(","))
    labelPointRDD = lines.map(lambda r:
                              LabeledPoint(
                                  extract_label(r),
                                  extract_features(r, len(r) - 1)))
    return labelPointRDD

def SplitData(labelPointRDD):
    (trainData, validationData, testData) = labelPointRDD.randomSplit([8,1,1])
    print("trainData: " + str(trainData.count()) + 
          " validationData: " + str(validationData.count()) +
          " testData: " + str(testData.count()))
    return (trainData, validationData, testData)
  
def TrainEvaluateModel(trainData,validationData,
                       impurityParm,maxDepthParm,maxBinsParm):
    startTime = time()
    model = DecisionTree.trainRegressor(trainData,
                         categoricalFeaturesInfo={}, impurity=impurityParm, maxDepth=maxDepthParm,
                         maxBins=maxBinsParm)
    RMSE = EvaluateModel(model, validationData)
    duration = time() - startTime
    print("Evaluate the model: use the params: " + \
         "impurity=" + str(impurityParm) + \
         " maxDepthParm=" + str(maxDepthParm) + \
         " maxBinsParm=" + str(maxBinsParm) + "\n" + \
         "====> duration time = " + str(duration) + \
         " result RMSE = " + str(RMSE))
    return (RMSE, duration, impurityParm, maxDepthParm, maxBinsParm, model)

def EvaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    score = score.map(lambda x: float(x))
    scoreAndLables = score.zip(validationData.map(lambda p: p.label))
    metric = RegressionMetrics(scoreAndLables)
    RMSE = metric.rootMeanSquaredError
    return(RMSE)
  
def showchart(df,evalparm,barData,lineData,yMin,yMax):
    ax = df[barData].plot(kind='bar',title=evalparm,
                         figsize=(10,6),legend=True,fontsize=12)
    ax.set_xlabel('RMSE',fontsize=12)
    ax.set_ylim([yMin,yMax])
    ax.set_ylabel('RMSE',fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[lineData].values, linestyle='-',marker='o',
            linewidth=2.0,color='r')
    plt.show()

def ElvaParameter(trainData, validationData, evalparm,
                  impurityList, maxDepthList, maxBinsList):
    metrics = [TrainEvaluateModel(trainData, validationData,
                                  impurity, maxDepth, maxBins)
               for impurity in impurityList
               for maxDepth in maxDepthList
               for maxBins in maxBinsList]

    if evalparm == "impurity":
        indexList = impurityList[:]
    elif evalparm == "maxDepth":
        indexList = maxDepthList[:]
    elif evalparm == "maxBins":
        indexList = maxBinsList[:]
    df = pd.DataFrame(metrics, index=indexList,
                      columns=['RMSE', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
    showchart(df, 'impurity', 'RMSE', 'duration', 0, 200)
    
def ParametersEval(trainData, validationData):
    ElvaParameter(trainData, validationData, "maxDepth",
                  impurityList=["variance"],maxDepthList=[3,5,10],maxBinsList=[10])
    ElvaParameter(trainData, validationData, "maxBins",
                  impurityList=["variance"],maxDepthList=[5],maxBinsList=[10,50,100])
    
def EvalAllParameter(trainData,validationData,
                 impurityList,maxDepthList,maxBinsList):
    metrics=[TrainEvaluateModel(trainData,validationData,
                                impurityParm,maxDepthParm,maxBinsParm)
            for impurityParm in impurityList
            for maxDepthParm in maxDepthList
            for maxBinsParm in maxBinsList]
    Smetrics = sorted(metrics, key=lambda k:k[0])
    bestParameter=Smetrics[0]
    print("the adjusted best params are: impurity: " + str(bestParameter[2]) +
                                      ", maxDepth: " + str(bestParameter[3]) +
                                      ", maxBins: " + str(bestParameter[4]) +
                                      "\nthe reuslt RMSE = " + str(bestParameter[0])
         )
    return bestParameter[5]
  
def PredictData(model, labelPointRDD):
    seasonDict = {1:'spring', 2:'summer', 3:'auttom', 4:'winter'}
    holidayDict = {0:'Not holiday', 1:"holiday"}
    weekDict = {0:'MON', 1:'TUES', 2:'WED',3: 'THURS', 4:'FRI',5:'SUN',6:'SAT'}
    workDayDict = {1:'Not working-day', 0:'working-day'}
    weatherDict = {1:'sunny', 2:'wet', 3:'rainy', 4:'intense fall'}
    
    for lp in labelPointRDD.take(20):
        predict = int(model.predict(lp.features))
        label = lp.label
        features = lp.features
        result = (" right " if label==predict else " false ")
        error = math.fabs(label - predict)
        print("characters: season: " + seasonDict[features[0]] + "\t" + 
                        "month: " + str(features[1]) + "\t" +
                        "hour: " + str(features[2]) + "\t" +
                        "holiday: " + holidayDict[features[3]] + "\t" +
                        "weekday:  " + weekDict[features[4]] + "\t" + 
                        "workingday: " + workDayDict[features[5]] + "\t" +
                        "weatherday: " + weatherDict[features[6]] + "\t" +
                        "temperature: " + str(features[7]) + "\t" +
                        "atemp: " + str(features[8]) + "\t" +
                        "humity: " + str(features[9]) + "\t" +
                        "windspeed: " + str(features[10]) +
                        "....==> predict: " + str(predict) + "\t" + 
                        "real: " + str(label) + "\t" +
                        "result: " + str(result) + "error: " + str(error))

if __name__ == "__main__":
    try:
        sc.stop()
    except:
        pass
    print("RunDecisionTreeMultiClass")
    sc = CreateSparkContext()
    print("==== Prepare Data =====")
    labelPointRDD = PrepareData(sc)
    (trainData, validationData, testData) = SplitData(labelPointRDD)
    trainData.persist(); validationData.persist(); testData.persist()
    print("==== access the train=====")
    (RMSE, duration, impurity, maxDepth, maxBins, model) = \
          TrainEvaluateModel(trainData,validationData,"variance",5,5)
    if (len(sys.argv) == 2) and (sys.argv[1] == "-e"):
        ParametersEval(trainData,validationData)
    if (len(sys.argv) == 2) and (sys.argv == "-a"):
        print("the best combination of params")
        model = EvalAllParameter(trainData, validationData,
                                 ["variance"],
                                 [3, 5, 10],
                                 [10, 50, 100])
    print("====== begin to train =========")
    RMSE = EvaluateModel(model,testData)
    print("Using the test data to test the best model , the result of RMSE is " + str(RMSE))
    print("=======begin to predict the data =======")
    PredictData(model, labelPointRDD)
