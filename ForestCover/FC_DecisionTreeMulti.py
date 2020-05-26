import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree


def SetPath(sc):
    global Path
    if sc.master[0:5] == "local":
        Path = "file:/usr/local/py_case/pj_UCI/"
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
    return float(label) - 1

def extract_features(record, featureEnd):
    numericalFeatures = [convert_float(field) for field in record[0: featureEnd]]
    return numericalFeatures

def PrepareData(sc):
    print("=========loading and transfer data===============")
    print("=========build RDD for training and accessing============")
    print("=========seperate three parts as trainData, validationData and train data=======")
    print("=========now begin to load the data=======")
    rawData = sc.textFile(Path + "data/covtype.data")
    rawData = rawData.take(5000)
    rawData = sc.parallelize(rawData)
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
    model = DecisionTree.trainClassifier(trainData,
                         numClasses=7, categoricalFeaturesInfo={}, impurity=impurityParm, maxDepth=maxDepthParm,
                         maxBins=maxBinsParm)
    accuracy = EvaluateModel(model, validationData)
    duration = time() - startTime
    print("Evaluate the model: use the params: " + \
         "impurity=" + str(impurityParm) + \
         " maxDepthParm=" + str(maxDepthParm) + \
         " maxBinsParm=" + str(maxBinsParm) + "\n" + \
         "====> duration time = " + str(duration) + \
         " result AUC = " + str(accuracy))
    return (accuracy, duration, impurityParm, maxDepthParm, maxBinsParm, model)

def EvaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    score = score.map(lambda x: float(x))
    scoreAndLables = score.zip(validationData.map(lambda p: p.label))
    metric = MulticlassMetrics(scoreAndLables)
    accuracy = metric.accuracy
    return(accuracy)

def showchart(df,evalparm,barData,lineData,yMin,yMax):
    ax = df[barData].plot(kind='bar',title=evalparm,
                         figsize=(10,6),legend=True,fontsize=12)
    ax.set_xlabel(evalparm,fontsize=12)
    ax.set_ylim([yMin,yMax])
    ax.set_ylabel(barData,fontsize=12)
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
                      columns=['accuracy', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
    showchart(df, 'impurity', 'accuracy', 'duration', 0.6, 1)
    
def ParametersEval(trainData, validationData):
    ElvaParameter(trainData,validationData,"impurity",
                  impurityList=["gini","entropy"],maxDepthList=[5],maxBinsList=[10])
    ElvaParameter(trainData, validationData, "maxDepth",
                  impurityList=["gini"],maxDepthList=[3,5,10],maxBinsList=[10])
    ElvaParameter(trainData, validationData, "maxBins",
                  impurityList=["gini"],maxDepthList=[5],maxBinsList=[10,50,100])



def EvalAllParameter(trainData,validationData,
                 impurityList,maxDepthList,maxBinsList):
    metrics=[TrainEvaluateModel(trainData,validationData,
                                impurityParm,maxDepthParm,maxBinsParm)
            for impurityParm in impurityList
            for maxDepthParm in maxDepthList
            for maxBinsParm in maxBinsList]
    Smetrics = sorted(metrics, key=lambda k:k[0],reverse=True)
    bestParameter=Smetrics[0]
    print("the adjusted best params are: impurity: " + str(bestParameter[2]) +
                                      ", maxDepth: " + str(bestParameter[3]) +
                                      ", maxBins: " + str(bestParameter[4]) +
                                      "\nthe reuslt AUC = " + str(bestParameter[0])
         )
    return bestParameter[5]

def PredictData(model, labelPointRDD):
    for lp in labelPointRDD.take(1000):
        predict = model.predict(lp.features)
        label = lp.label
        features = lp.features
        result = (" right " if label==predict else " false ")
        print("ground situation: altutde: " + str(features[0]) + "\t" + 
                        "position: " + str(features[1]) + "\t" +
                        "gradient: " +str(features[2]) + "\t" +
                        "vertival distance of water source: " + str(features[3]) + "\t" +
                        "horisontal distance of water source: " + str(features[4]) + "\t" + 
                        "shadow at 9 o'clock: " +str(features[5]) + "/t" + 
                        "....==> predict: " + str(predict) + "/t" + 
                        "real: " + str(label) + "/t" +
                        "result: " + str(result))
        
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
    (accuracy, duration, impurity, maxDepth, maxBins, model) = \
          TrainEvaluateModel(trainData,validationData,"entropy",5,5)
    if (len(sys.argv) == 2) and (sys.argv[1] == "-e"):
        ParametersEval(trainData,validationData)
    if (len(sys.argv) == 2) and (sys.argv == "-a"):
        print("the best combination of params")
        model = EvalAllParameter(trainData, validationData,
                                 ["gini", "entropy"],
                                 [3, 5, 10],
                                 [10, 50, 100])
    print("====== begin to train =========")
    accuracy = EvaluateModel(model,testData)
    print("Using the test data to test the best model , the result of accuracy is " + str(accuracy))
    print("=======begin to predict the data =======")
    PredictData(model, labelPointRDD)
