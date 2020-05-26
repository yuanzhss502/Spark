import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree


def SetPath(sc):
    global Path
    if sc.master[0:5] == "local":
        Path = "file:/usr/local/py_case/pj_SE/"
    else:
        Path = "hdfs://master:9000/py_case/PythonProject_2/"


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

def extract_label(field):
    label = (field[-1])
    return float(label)


def extract_features(field, categoriesMap, featureEnd):
    categoryIdx = categoriesMap[field[3]]
    categoryFeatures = np.zeros(len(categoriesMap))
    categoryFeatures[categoryIdx] = 1
    numericalFeatures = [convert_float(field) for field in field[4:featureEnd]]
    return np.concatenate((categoryFeatures, numericalFeatures))


def convert_float(x):
    return (0 if x == "?" else float(x))

def PrepareData(sc):
    rawDataWithHeader = sc.textFile(Path + "data/train.tsv")
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    rData = rawData.map(lambda x: x.replace("\"", ""))
    lines = rData.map(lambda x: x.split("\t"))
    print("total " + str(lines.count()))
    print("=======before standare========")
    categoriesMap = lines.map(lambda fields: fields[3]) \
        .distinct() \
        .zipWithIndex().collectAsMap()
    labelRDD = lines.map(lambda r: extract_label(r))
    featureRDD = lines.map(lambda r: extract_features(r, categoriesMap, len(r)-1))
    for i in featureRDD.first():
        print(str(i) + ", ")
    print("=======after standare========")
    stdScale = StandardScaler(withMean=True, withStd=True).fit(featureRDD)
    scaleFeatureRDD = stdScale.transform(featureRDD)
    for i in scaleFeatureRDD.first():
        print(str(i) + ",")
    labelPoint = labelRDD.zip(scaleFeatureRDD)
    labelPointRDD = labelPoint.map(lambda r: LabeledPoint(r[0],r[1]))
    (trainData, validationData, testData) = labelPointRDD.randomSplit([8, 1, 1])
    return (trainData, validationData, testData, categoriesMap)

def EvaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    # 这里的score是int，要转换为float
    score = score.map(lambda x: float(x))
    scoreAndLables = score.zip(validationData.map(lambda p: p.label))
    metric = BinaryClassificationMetrics(scoreAndLables)
    AUC = metric.areaUnderROC
    return(AUC)

def TrainEvaluateModel(trainData,validationData,
                       numIterations,stepSize,regParam):
    startTime = time()
    model = SVMWithSGD.train(trainData,
                         numIterations,stepSize,regParam)
    AUC = EvaluateModel(model, validationData)
    duration = time() - startTime
    print("Evaluate the model: use the params: " + \
         "numIterations=" + str(numIterations) + \
         " stepSize" + str(stepSize) + \
         " regParam=" + str(regParam) + "\n" + \
         "====> duration time = " + str(duration) + \
         " result AUC = " + str(AUC))
    return (AUC, duration, numIterations, stepSize, regParam, model)

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
                  numIterationsList, stepSizeList, regParamList):
    metrics = [TrainEvaluateModel(trainData, validationData,
                                  numIterations, stepSize, regParam)
               for numIterations in numIterationsList
               for stepSize in stepSizeList
               for regParam in regParamList]

    if evalparm == "numIterations":
        indexList = numIterationsList[:]
    elif evalparm == "stepSize":
        indexList = stepSizeList[:]
    elif evalparm == "regParam":
        indexList = regParamList[:]
    df = pd.DataFrame(metrics, index=indexList,
                      columns=['AUC', 'duration', 'numIterations', 'stepSize', 'regParam', 'model'])
    showchart(df, evalparm, 'AUC', 'duration', 0.5, 0.7)

def ParametersEval(trainData, validationData):
    ElvaParameter(trainData,validationData,"numIterations",
                  numIterationsList=[1,3,5,15,25],stepSizeList=[100],regParamList=[1])
    ElvaParameter(trainData, validationData, "stepSize",
                  numIterationsList=[25],stepSizeList=[10,50,100,200],regParamList=[1])
    ElvaParameter(trainData, validationData, "regParm",
                  numIterationsList=[25],stepSizeList=[100],regParamList=[0.01,0.1,1])



def EvalAllParameter(trainData,validationData,
                  numIterationsList, stepSizeList, regParamList):
    metrics=[TrainEvaluateModel(trainData,validationData,
                                 numIterations, stepSize, regParamtion)
            for numIterations in numIterationsList
            for stepSize in stepSizeList
            for regParam in regParamList]
    Smetrics = sorted(metrics, key=lambda k:k[0],reverse=True)
    bestParameter=Smetrics[0]
    print("the adjusted best params are: numIterations: " + str(bestParameter[2]) +
                                      ", stepSize: " + str(bestParameter[3]) +
                                      ", regParam: " + str(bestParameter[4]) +
                                      "\nthe reuslt AUC = " + str(bestParameter[0])
         )
    return bestParameter[5]


def PredictData(sc, model, categoriesMap):
    print("begin to load the text")
    rawDataWithHeader = sc.textFile(Path + 'data/test.tsv')
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    rData = rawData.map(lambda x: x.replace("\"", ""))
    lines = rData.map(lambda x: x.split("\t"))
    print("total " + str(lines.count()))
    dataRDD = lines.map(lambda r: (r[0],
                                   extract_features(r, categoriesMap, len(r))))
    DescDict = {
        0: "ephemeral website",
        1: "evergreen website"

    }
    for data in dataRDD.take(10):
        predictResult = model.predict(data[1])
        print("website: " + str(data[0]) + "\n" + "==> predict " + str(predictResult) + "illustration " + DescDict[
            predictResult] + "\n")

if __name__ == "__main__":
    try:
        sc.stop()
    except:
        pass
    print("RunSVMWithSGDBinary")
    sc = CreateSparkContext()
    print("==== Prepare Data =====")
    (trainData, valiationData, testData, categoriesMap) = PrepareData(sc)
    trainData.persist(); valiationData.persist(); testData.persist()
    print("==== access the train")
    (AUC, duration, numIterations, stepSize, miniBatchFraction, model) = \
        TrainEvaluateModel(trainData,valiationData,3,50,1)
    if (len(sys.argv) == 2) and (sys.argv[1] == "-e"):
        ParametersEval(trainData,valiationData)
    if (len(sys.argv) == 2) and (sys.argv == "-a"):
        print("the best combination of params")
        model = EvalAllParameter(trainData, valiationData,
                                 [1,3,5,15,25],
                                 [10,50,100,200],
                                 [0.01,0.1,1])
    print("====== begin to train =========")
    auc = EvaluateModel(model,testData)
    print("Using the test data to test the best model , the result of AUC is " + str(auc))
    print("=======begin to predict the data =======")
    PredictData(sc, model, categoriesMap)
