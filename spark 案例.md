## 案例一（python spark 创建推荐引擎）

### 		1.1 数据准备

* 创建项目文件夹，并下载所需数据文件

  sudo mkdir -p /usr/local/PythonProject_1/data

  cd /usr/local/usr/local/PythonProject_1/data

  weget http://files.grouplens.org/datasets/movielens/ml-100k.zip

* 解压ml-100k

  unzip -j ml-100k

* 创建输入输出文件夹

  hdfs fs -mkdir -p /py_case/PythonProject_1/data

* 把数据文件导入到hdfs中
  hdfs fs copyFromLocal -f /usrlocal/PythonProject_1/data /pycase/PythonProject_1/



### 2.1 创建RecommendTrain.py 模型程序代码

```python
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.recommendation import Rating
from pyspark.mllib.recommendation import ALS


def SetPath(sc):
    global Path
    if sc.master[0:5] == "local":
        Path = "file:/usr/local/PythonProject_1/"
    else:
        Path = "hdfs://master:9000/py_case/PythonProject_1/"


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

def PrepareData(sc):
    rawUserData = sc.textFile(Path + "data/u.data")
    rawRatings = rawUserData.map(lambda line: line.split("\t")[:3])
    ratingsRDD = rawRatings.map(lambda x: (x[0], x[1], x[2]))
    return ratingsRDD

def SaveModel(sc):
    try:
        model.save(sc, Path + "ALSmodel")
        print("The model has been saved in ALSmodel")
    except Exception:
        print("The model has been existed, please delete it first.")

if __name__ == "__main__":
    sc = CreateSparkContext()
    print("========= the process of preparing data ===============")
    ratingsRDD = PrepareData(sc)
    print("===============the process of training data============")
    model = ALS.train(ratingsRDD, 10, 10, 0.01)
    print(model)
    print("==================save the model=======================")
    SaveModel(sc)
    
```

### 		3.1 创建Recommend.py推荐程序代码

```python
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.recommendation import MatrixFactorizationModel
import sys

def SetPath(sc):
    global Path
    if sc.master[0:5] == "local":
        Path = "file:/usr/local/PythonProject_1/"
    else:
        Path = "hdfs://master:9000/py_case/PythonProject_1/"


def SetLogger(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)


def CreateSparkContext():
    sparkConf = SparkConf() \
        .setAppName("Recommend") \
        .set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf=sparkConf)
    SetPath(sc)
    SetLogger(sc)
    print("master=" + sc.master)
    return sc

def PrepareData(sc):
    itemRDD = sc.textFile(Path + "data/u.item")
    movieTitle = itemRDD.map(lambda line: line.split("|"))  \
                        .map(lambda a : (float(a[0]),a[1])) \
                        .collectAsMap()
    return movieTitle

def loadModel(sc):
    try:
        model = MatrixFactorizationModel.load(sc, Path + "ALSmodel")
        print("begin to load the model")
    except Exception:
        print("we can not find the model, please train the data first")
    return model

def Recommend(model):
    if sys.argv[1] == "__U":
        RecommendMovies(model, movieTitle, int(sys.argv[2]))
    if sys.argv[1] == "__M":
        RecommendUsers(model, movieTitle, int(sys.argv[2]))

def RecommendMovies(model, movieTitle, inputUserID):
    recommendMovie = model.recommendProducts(inputUserID, 10)
    print("To the user id " + str(inputUserID) + "we recommend 10 movies as fllow:")
    for rmd in recommendMovie:
        print("To the user {0} recommend movie {1} the rating is {2}" \
                    .format(rmd[0], movieTitle[rmd[1]], rmd[2] ) )

def RecommendUsers(model, movieTitle, inputMovieID):
    recommendUser = model.recommendUsers(inputMovieID, 10)
    print("The the movie id {0} movie title {1} recommend the following users" \
                    .format(inputMovieID, movieTitle[inputMovieID]))
    for rmd in recommendUser:
        print("To the user id {0} rating is {1}".format(rmd[0], rmd[2]))


if __name__ == "__main__":
    if len(sys.argv) !=3:
        print("please enter two params")
        exit(-1)
    sc = CreateSparkContext()
    print("=============begin to prepare data===========")
    movieTitle = PrepareData(sc)
    print("=============begin to load the model============")
    model = loadModel(sc)
    print("=============according to the model, begin to recommend==============")
    Recommend(model)

```

### 	4.1 运行程序

* 运行RecommendTrain.py 创建并保存模型

  ```Linux
  HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop spark-submit --driver-memory 512m --executor-cores 2 --master yarn --deploy-mode  client RecommendTrain.py 
  ```

* 运行Recommend.py 针对ID 100 的用户推荐10部电影

  ```Linux
  HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop/ spark-submit --driver-memory 512m --executor-cores 2 --executor-memory 1g --master yarn --deploy-mode client /usr/local/PythonProject_1/Recommend.py __U 100
  ```

* 运行Recommend\.py 针对ID 100 的电影 推荐给10名用户

  ```Linux
  HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop/ spark-submit --driver-memory 512m --executor-cores 2 --executor-memory 1g --master yarn --deploy-mode client /usr/local/PythonProject_1/Recommend.py __M 100
  ```



## 案例二：("StumbleUpon Evergreen" 大数据问题)

### 		2.1. 使用ipython进行开发

* 为什么要使用ipthon进行数据分析

  ipython对代码进行分段处理，数据处理过程中，能对处理后的数据进行观察，一般是进行本地开发

  这样数据处理得更快占用更少内存，开发完成后再使用其他模式进行运行

```Linux
PYSPARK_DRIVER_PYTHON=ipython PYSPARK_DRIVER_PYTHON_OPTS="notebook" MASTER=local[*] pyspark
```

### 	2.2 数据准备

```python
global Path
if sc.master[0:5]=="local":
    Path="file:/usr/local/py_case/pj_SE/"
else:
    Path="hdfs://master:9000/py_case/PythonProject_2/"    
```

* 加载数据并进行数据清洗

  **原数据：**

  ['"url"\t"urlid"\t"boilerplate"\t"alchemy_category"\t"alchemy_category_score"\t"avglinksize"\t"commonlinkratio_1"\t"commonlinkratio_2"\t"commonlinkratio_3"\t"commonlinkratio_4"\t"compression_ratio"\t"embed_ratio"\t"framebased"\t"frameTagRatio"\t"hasDomainLink"\t"html_ratio"\t"image_ratio"\t"is_news"\t"lengthyLinkDomain"\t"linkwordscore"\t"news_front_page"\t"non_markup_alphanum_characters"\t"numberOfLinks"\t"numwords_in_url"\t"parametrizedLinkRatio"\t"spelling_errors_ratio"\t"label"']

  ```python
  print("begin to load the data")
  rawDataWithHeader = sc.textFile(Path + "data/train.tsv")
  header = rawDataWithHeader.first()
  rawData = rawDataWithHeader.filter(lambda x: x != header)
  rData = rawData.map(lambda x: x.replace("\"", ""))
  lines = rData.map(lambda x: x.split("\t"))
  print("total " + str(lines.count()))
  ```

  * 清洗思路
    * 去除表头
    * 去除字段前后的双引号""
    * 字段间使用制表符\t分割

* 处理分类特征字段(如bussiness healthy 等)

  ```python
  categoriesMap = lines.map(lambda fields : fields[3]) \
                      .distinct() \
                      .zipWithIndex().collectAsMap()
  ```

  * 处理思路

    * 去重
    * 对不重复的分类进行编号
    * 转换为dict类型

    **结果：** categoriesMap

    {'business': 0, 'recreation': 1, 'health': 2, 'sports': 3, '?': 4, 'arts_entertainment': 5, 'science_technology': 6, 'gaming': 7, 'culture_politics': 8, 'computer_internet': 9, 'law_crime': 10, 'religion': 11, 'weather': 12, 'unknown': 13}

* 提取分类特征字段和数值特征字段

  ```python
  import numpy as np
  
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
      return (0 if x=="?" else float(x))
      
  from pyspark.mllib.regression import LabeledPoint
  labelpointRDD = lines.map(lambda r:
          LabeledPoint(
              extract_label(r),
              extract_features(r, categoriesMap, len(r)-1)))
  
  ```

  * 处理思路

    * 先提取label(虽然可以观察得到label有14个 为了更精准，更新更及时应该进行机器提取)
    * 提取有用特征字段(从第四行开始即field[3])
      * 把分类特征字段转换为数值字段，通过np创建空间为14的列表，根据分类编号在对应位置填充1
      * 当字段出现？时，用0代替
      * 把数值特征字段改变为float类型
      * 合并分类特征字段和数值特征字段

    **结果：**labelpointRDD

    [LabeledPoint(0.0, [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.789131,2.055555556,0.676470588,0.205882353,0.047058824,0.023529412,0.443783175,0.0,0.0,0.09077381,0.0,0.245831182,0.003883495,1.0,1.0,24.0,0.0,5424.0,170.0,8.0,0.152941176,0.079129575])]

* 以随机方式把数据分为三部分 8:1:1

  ```python
  (trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
  print(
      "The data has been seperated as three parts, there are " + str(trainData.count()) + 
      " train data " + str(validationData.count()) + " validationData " + 				str(testData.count()) +" train data"
  )
  
  trainData.persist()
  validationData.persist()
  testData.persist()
  ```

  

###   2.3 建立模型(DescionTree)

```python
from pyspark.mllib.tree import DecisionTree
model = DecisionTree.trainClassifier( \
                trainData, numClasses=2, categoricalFeaturesInfo={}, \
                impurity="entropy", maxDepth=5, maxBins=5
                                    )
```



###   2.4 预测数据

```python
def PredictData(sc,model,categoriesMap):
    print("begin to load the text")
    rawDataWithHeader = sc.textFile(Path + 'data/test.tsv')
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    rData = rawData.map(lambda x: x.replace("\"",""))
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
        print("website: " + str(data[0]) + "\n" + "==> predict " + str(predictResult) + " illustration " + DescDict[predictResult] + "\n")

print("======predict the data =========")
PredictData(sc,model,categoriesMap)
```

* **结果**

  ======predict the data =========
  begin to load the text
  total 3171
  website: http://www.lynnskitchenadventures.com/2009/04/homemade-enchilada-sauce.html
  ==> predict 1.0illustration evergreen website

  website: http://lolpics.se/18552-stun-grenade-ar
  ==> predict 0.0illustration ephemeral website

  website: http://www.xcelerationfitness.com/treadmills.html
  ==> predict 0.0illustration ephemeral website

  website: http://www.bloomberg.com/news/2012-02-06/syria-s-assad-deploys-tactics-of-father-to-crush-revolt-threatening-reign.html
  ==> predict 0.0illustration ephemeral website

  website: http://www.wired.com/gadgetlab/2011/12/stem-turns-lemons-and-limes-into-juicy-atomizers/
  ==> predict 0.0illustration ephemeral website

  website: http://www.latimes.com/health/boostershots/la-heb-fat-tax-denmark-20111013,0,2603132.story
  ==> predict 0.0illustration ephemeral website

  website: http://www.howlifeworks.com/a/a?AG_ID=1186&cid=7340ci
  ==> predict 1.0illustration evergreen website

  website: http://romancingthestoveblog.wordpress.com/2010/01/13/sweet-potato-ravioli-with-lemon-sage-brown-butter-sauce/
  ==> predict 1.0illustration evergreen website

  website: http://www.funniez.net/Funny-Pictures/turn-men-down.html
  ==> predict 0.0illustration ephemeral website



### 	2.5 评估模型的准确率

```python
from pyspark.mllib.evaluation import BinaryClassificationMetrics

def EvaluateModel(model, valiationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLables = score.zip(validationData.map(lambda p: p.label))
    metric = BinaryClassificationMetrics(scoreAndLables)
    AUC = metric.areaUnderROC
    return(AUC)
```

* 思路分析
  * trainData用于建立模型，valiationData用于使用
  * 提取valiationData的feature部分，然后根据14项features对数据进行分析
  * 得出结果(1 or 0)
  * 把预测结果与正确结果结合在一起(zip) 
  * 通过预测结果与正确结果计算出ROC

###   2.6 模型的训练参数如何影响准确率

* 建立 trainEvaluateModel

  ```python
  from time import time
  def TrainEvaluateModel(trainData,validationData,
                         impurityParm,maxDepthParm,maxBinsParm):
      startTime = time()
      model = DecisionTree.trainClassifier(trainData, 
                           numClasses=2, categoricalFeaturesInfo={}, impurity=impurityParm, maxDepth=maxDepthParm,
                           maxBins=maxBinsParm)
      AUC = EvaluateModel(model, validationData)
      duration = time() - startTime
      print("Evaluate the model: use the params: " + \
           "impurity=" + str(impurityParm) + \
           " maxDepthParm=" + str(maxDepthParm) + \
           " maxBinsParm=" + str(maxBinsParm) + "\n" + \
           "====> duration time = " + str(duration) + \
           " result AUC = " + str(AUC))
      return (AUC, duration, impurityParm, maxDepthParm, maxBinsParm, model)
    
  impurityList = ["gini", "entropy"]
  maxDepthList = [10]
  maxBinsList = [10]
  
  metrics = [TrainEvaluateModel(trainData, validationData, impuity, maxDepthParm, maxBinsParm)
            for impuity in impurityList
            for maxDepthParm in maxDepthList
            for maxBinsParm in maxBinsList]
  ```

* **结果**

  Evaluate the model: use the params: impurity=gini maxDepthParm=10 maxBinsParm=10
  ====> duration time = 2.6519758701324463 result AUC = 0.6524691977796584
  Evaluate the model: use the params: impurity=entropy maxDepthParm=10 maxBinsParm=10
  ====> duration time = 1.9317045211791992 result AUC = 0.6367776359385533

###   2.7 训练的结果以图表显示

* 用pandas模块对训练的结果以表格的形式进行展示

  ```python
  import pandas as pd
  IndexList = impurityList
  df = pd.DataFrame(metrics,index=IndexList,
          columns=['AUC', 'duration', 'impuity', 'maxDepth', 'maxBins', 'model'])
  df
  ```

* 用Matplotlib模块对训练的结果以图表的形式进行展示

  ```python
  import matplotlib.pyplot as plt
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
  ```

###   2.8 完整程序

```python
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyspark import SparkConf
from pyspark import SparkContext
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
    categoriesMap = lines.map(lambda fields: fields[3]) \
        .distinct() \
        .zipWithIndex().collectAsMap()
    labelpointRDD = lines.map(lambda r:
                              LabeledPoint(
                                  extract_label(r),
                                  extract_features(r, categoriesMap, len(r) - 1)))
    (trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
    return (trainData, validationData, testData, categoriesMap)

def EvaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLables = score.zip(validationData.map(lambda p: p.label))
    metric = BinaryClassificationMetrics(scoreAndLables)
    AUC = metric.areaUnderROC
    return(AUC)

def TrainEvaluateModel(trainData,validationData,
                       impurityParm,maxDepthParm,maxBinsParm):
    startTime = time()
    model = DecisionTree.trainClassifier(trainData,
                         numClasses=2, categoricalFeaturesInfo={}, impurity=impurityParm, maxDepth=maxDepthParm,
                         maxBins=maxBinsParm)
    AUC = EvaluateModel(model, validationData)
    duration = time() - startTime
    print("Evaluate the model: use the params: " + \
         "impurity=" + str(impurityParm) + \
         " maxDepthParm=" + str(maxDepthParm) + \
         " maxBinsParm=" + str(maxBinsParm) + "\n" + \
         "====> duration time = " + str(duration) + \
         " result AUC = " + str(AUC))
    return (AUC, duration, impurityParm, maxDepthParm, maxBinsParm, model)

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
                      columns=['AUC', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
    showchart(df, 'impurity', 'AUC', 'duration', 0.5, 0.7)

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
    print("RunDecisionTreeBinary")
    sc = CreateSparkContext()
    print("==== Prepare Data =====")
    (trainData, valiationData, testData, categoriesMap) = PrepareData(sc)
    trainData.persist(); valiationData.persist(); testData.persist()
    print("==== access the train")
    (AUC, duration, impurity, maxDepth, maxBins, model) = \
        TrainEvaluateModel(trainData,valiationData,"entropy",5,5)
    if (len(sys.argv) == 2) and (sys.argv[1] == "-e"):
        ParametersEval(trainData,valiationData)
    if (len(sys.argv) == 2) and (sys.argv == "-a"):
        print("the best combination of params")
        model = EvalAllParameter(trainData, valiationData,
                                 ["gini", "entropy"],
                                 [3, 5, 10],
                                 [10, 50, 100])
    print("====== begin to train =========")
    auc = EvaluateModel(model,testData)
    print("Using the test data to test the best model , the result of AUC is " + str(auc))
    print("=======begin to predict the data =======")
    PredictData(sc, model, categoriesMap)
    print(model.toDebugString())
```



###   3.1 使用逻辑回归二元分类

* 使用多元回归分析构建模型(LogisiticRegressionWithSGDBinary)

  ```python
  # 用于
  from pyspark.mllib.classification import LogisticRegressionWithSGD
  # 用于对数值特征的字段进行标准化处理
  from pyspark.mllib.feature import StandardScaler
  ```

###   3.2  对数值特征字段进行标准化处理

```python
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
```

* 处理思路
  * 创建标准化刻度，并引入withMean和withStd参数
  * 进行标准化转换
  * 把标签label和标准化后的特征数据scaleFeatureRDD整合在一起
  * 创建LabelPoint数据



###   3.3 训练模型

* 使用numIterations, stepSize 和 miniBatchFraction参数

  ```python
  def TrainEvaluateModel(trainData,validationData,
                         numIterations,stepSize,miniBatchFraction):
      startTime = time()
      model = LogisticRegressionWithSGD.train(trainData,
                           numIterations,stepSize,miniBatchFraction)
      AUC = EvaluateModel(model, validationData)
      duration = time() - startTime
      print("Evaluate the model: use the params: " + \
           "numIterations=" + str(numIterations) + \
           " stepSize" + str(stepSize) + \
           " miniBatchFraction=" + str(miniBatchFractionm) + "\n" + \
           "====> duration time = " + str(duration) + \
           " result AUC = " + str(AUC))
      return (AUC, duration, numIterations, stepSize, miniBatchFraction, model)
  ```

  

###   3.4 完整程序

```python
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithSGD
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
                       numIterations,stepSize,miniBatchFraction):
    startTime = time()
    model = LogisticRegressionWithSGD.train(trainData,
                         numIterations,stepSize,miniBatchFraction)
    AUC = EvaluateModel(model, validationData)
    duration = time() - startTime
    print("Evaluate the model: use the params: " + \
         "numIterations=" + str(numIterations) + \
         " stepSize" + str(stepSize) + \
         " miniBatchFraction=" + str(miniBatchFraction) + "\n" + \
         "====> duration time = " + str(duration) + \
         " result AUC = " + str(AUC))
    return (AUC, duration, numIterations, stepSize, miniBatchFraction, model)

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
                  numIterationsList, stepSizeList, miniBatchFractionList):
    metrics = [TrainEvaluateModel(trainData, validationData,
                                  numIterations, stepSize, miniBatchFraction)
               for numIterations in numIterationsList
               for stepSize in stepSizeList
               for miniBatchFraction in miniBatchFractionList]

    if evalparm == "numIterations":
        indexList = numIterationsList[:]
    elif evalparm == "stepSize":
        indexList = stepSizeList[:]
    elif evalparm == "miniBatchFraction":
        indexList = miniBatchFractionList[:]
    df = pd.DataFrame(metrics, index=indexList,
                      columns=['AUC', 'duration', 'numIterations', 'stepSize', 'miniBatchFraction', 'model'])
    showchart(df, evalparm, 'AUC', 'duration', 0.5, 0.7)

def ParametersEval(trainData, validationData):
    ElvaParameter(trainData,validationData,"numIterations",
                  numIterationsList=[5,20,60,100],stepSizeList=[10],miniBatchFractionList=[1])
    ElvaParameter(trainData, validationData, "stepSize",
                  numIterationsList=[100],stepSizeList=[10,50,100,200],miniBatchFractionList=[1])
    ElvaParameter(trainData, validationData, "miniBatchFraction",
                  numIterationsList=[100],stepSizeList=[100],miniBatchFractionList=[0.5,0.8,1])



def EvalAllParameter(trainData,validationData,
                  numIterationsList, stepSizeList, miniBatchFractionList):
    metrics=[TrainEvaluateModel(trainData,validationData,
                                 numIterations, stepSize, miniBatchFraction)
            for numIterations in numIterationsList
            for stepSize in stepSizeList
            for miniBatchFraction in miniBatchFractionList]
    Smetrics = sorted(metrics, key=lambda k:k[0],reverse=True)
    bestParameter=Smetrics[0]
    print("the adjusted best params are: numIterations: " + str(bestParameter[2]) +
                                      ", stepSize: " + str(bestParameter[3]) +
                                      ", miniBatchFraction: " + str(bestParameter[4]) +
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
    print("RunLogisiticRegressionWithSGDBinary")
    sc = CreateSparkContext()
    print("==== Prepare Data =====")
    (trainData, valiationData, testData, categoriesMap) = PrepareData(sc)
    trainData.persist(); valiationData.persist(); testData.persist()
    print("==== access the train")
    (AUC, duration, numIterations, stepSize, miniBatchFraction, model) = \
        TrainEvaluateModel(trainData,valiationData,15,10,0.5)
    if (len(sys.argv) == 2) and (sys.argv[1] == "-e"):
        ParametersEval(trainData,valiationData)
    if (len(sys.argv) == 2) and (sys.argv == "-a"):
        print("the best combination of params")
        model = EvalAllParameter(trainData, valiationData,
                                 [3,5,10,15],
                                 [10,50,100],
                                 [0.5,0.8,1])
    print("====== begin to train =========")
    auc = EvaluateModel(model,testData)
    print("Using the test data to test the best model , the result of AUC is " + str(auc))
    print("=======begin to predict the data =======")
    PredictData(sc, model, categoriesMap)
```



###   4.1 支持向量机SVM二元分类

* 使用SVMWithSGDBinary创建模型

  ```python
  from pyspark.mllib.classification import SVMWithSGD
  ```

###   4.2 对数值特征字段进行标准化处理

###   4.3 创建模型(SVMWithSGDBinary)

* 使用numIterations, stepSize, regParam 参数构建模型

  ```python
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
           " miniBatchFraction=" + str(regParam) + "\n" + \
           "====> duration time = " + str(duration) + \
           " result AUC = " + str(AUC))
      return (AUC, duration, numIterations, stepSize, regParam, model)
  ```

###   4.4 完整程序

```python
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
```



## 案例三：("森林覆盖植被"大数据问题分析)

| 字段        | 说明                                                         |
| :---------- | ------------------------------------------------------------ |
| feature特征 | Elvation、Aspect、Slope、vertical distance of water source、horizonal distance of water source、shadow at 9 o'clock |
| label标签   | Cover Type: 1.Spruce/Fir  2.Lodgepole Pine 3.Ponderosa Pine 4.Cottonwood/Willow 5.Aspen 6.Douglas-fir 7.Krummholz |

###   1.1 准备数据

```python
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
```

* 使用思路

  * 数据量比较多，可以只提取一部分数据测试

    * ```python
      # take使得rawData转化成list
      rawData = rawData.take(5000)
      # 再把list转回RDD
      rawData = sc.parallelize(rawData)
      ```

  * 数据间用","分割

  * 提取标签(最后一个字段),提取 特征值(除了最后的字段其余的字段)

###   2.1 创建模型(DecisionTreeMulti)

```python
def TrainEvaluateModel(trainData,validationData,
                       impurityParm,maxDepthParm,maxBinsParm):
    startTime = time()
    model = DecisionTree.trainClassifier(trainData,
                        numClasses=7, categoricalFeaturesInfo={}, 
                        impurity=impurityParm, maxDepth=maxDepthParm,  maxBins=maxBinsParm)
    accuracy = EvaluateModel(model, validationData)
    duration = time() - startTime
    print("Evaluate the model: use the params: " + \
         "impurity=" + str(impurityParm) + \
         " maxDepthParm=" + str(maxDepthParm) + \
         " maxBinsParm=" + str(maxBinsParm) + "\n" + \
         "====> duration time = " + str(duration) + \
         " result accuracy = " + str(accuracy))
    return (accuracy, duration, impurityParm, maxDepthParm, maxBinsParm, model)

```

* 使用思路
  * 与二元决策树不同的是这里需要预测的标签有7个，因此numClasses=7

###   3.1 评估最优模型

* 与案例二评估的方法一样

###   4.1 根据最优模型预测数据

```python
def PredictData(sc, model, labelPointRDD):
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
```

* 设计思路
  * 在PrepareData中经过提取标签和特征值经过LabelPointed处理过得到的labelPointRDD拥有label和feature的属性，如上例：lp.label lp.feature
  * 通过模型使用lp.feature数据得到预测结果predict
  * 对label和feature进行重新赋值
  * 检验预测结果是否与实际一样 predict == result?

###    5.1 完成程序

```python
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
```



## 案例四：（"BikeSharing"大数据问题分析）

| 字段         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| features特征 | Season、month、hour、holiday、week、working-day、weather、temperature、atemp、humity、windspeed |
| label标签    | casual + registered = cnt                                    |

###   1.1 使用决策二元回归

```python
import math
from pyspark.mllib.evaluation import RegressionMetrics
```

###    2.1创建模型

```python
    model = DecisionTree.trainRegressor(trainData,
                         categoricalFeaturesInfo={}, 
                         impurity=impurityParm, maxDepth=maxDepthParm, maxBins=maxBinsParm)
```

###   3.1 使用RMSE评估模型

```python
def EvaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    score = score.map(lambda x: float(x))
    scoreAndLables = score.zip(validationData.map(lambda p: p.label))
    metric = RegressionMetrics(scoreAndLables)
    RMSE = metric.rootMeanSquaredError
    return(RMSE)
```

###   4.1 完整程序

```python
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
```

###   5.1 预测结果

characters: season: spring	month: 1.0	hour: 0.0	holiday: Not holiday	weekday:  SAT	workingday: working-day	weatherday: sunny	temperature: 0.24	atemp: 0.2879	humity: 0.81	windspeed: 0.0....==> predict: 32	real: 16.0	result:  false error: 16.0
characters: season: spring	month: 1.0	hour: 1.0	holiday: Not holiday	weekday:  SAT	workingday: working-day	weatherday: sunny	temperature: 0.22	atemp: 0.2727	humity: 0.8	windspeed: 0.0....==> predict: 32	real: 40.0	result:  false error: 8.0
characters: season: spring	month: 1.0	hour: 2.0	holiday: Not holiday	weekday:  SAT	workingday: working-day	weatherday: sunny	temperature: 0.22	atemp: 0.2727	humity: 0.8	windspeed: 0.0....==> predict: 24	real: 32.0	result:  false error: 8.0
characters: season: spring	month: 1.0	hour: 3.0	holiday: Not holiday	weekday:  SAT	workingday: working-day	weatherday: sunny	temperature: 0.24	atemp: 0.2879	humity: 0.75	windspeed: 0.0....==> predict: 12	real: 13.0	result:  false error: 1.0
characters: season: spring	month: 1.0	hour: 4.0	holiday: Not holiday	weekday:  SAT	workingday: working-day	weatherday: sunny	temperature: 0.24	atemp: 0.2879	humity: 0.75	windspeed: 0.0....==> predict: 3	real: 1.0	result:  false error: 2.0



## 五. SparkSQL、DataFrame、RDD数据可视化

###   1.1 创建RDD、DataFrame 与 Spark SQL

```python
from pyspark.sql import Row

global Path
if sc.master[0:5]=='local':
    Path = '/usr/local/PythonProject_1'
else:
    Path = 'hdfs://master:9000/user/yuanzhss'

RawUserRDD = sc.textFile(Path + '/data/u.user')
userRDD = RawUserRDD.map(lambda line: line.split('|'))
sqlContext = SparkSession.builder.getOrCreate()
user_Rows = userRDD.map(lambda p:
                       Row(
                           userid=int(p[0]),
                           age=int(p[1]),
                           gender=str(p[2]),
                           occupation=str(p[3]),
                           zipcode=p[4],
                       ) )
user_df = sqlContext.createDataFrame(user_Rows)
user_df.printSchema()
user_df.show(5)
```

* RawUserRDD.take(3)
  * ['1|24|M|technician|85711',
    '2|53|F|other|94043',
    '3|23|M|writer|32067',]
* userRDD.take(3)
  * [['1', '24', 'M', 'technician', '85711'],
    ['2', '53', 'F', 'other', '94043'],
    ['3', '23', 'M', 'writer', '32067'],]
* user_Rows.take(3)
  * [Row(age=24, gender='M', occupation='technician', userid=1, zipcode='85711'),
    Row(age=53, gender='F', occupation='other', userid=2, zipcode='94043'),
    Row(age=23, gender='M', occupation='writer', userid=3, zipcode='32067')]
* user_df.printSchema()
  * root
    |-- age: long (nullable = true)
    |-- gender: string (nullable = true)
    |-- occupation: string (nullable = true
* user_df.show(3)
  * +---+------+----------+------+-------+
    |age|gender|occupation|userid|zipcode|
    +---+------+----------+------+-------+
    | 24|     M|technician|     1|  85711|
    | 53|     F |     other|     2|  94043|
    | 23|     M|    writer|     3|  32067|

​               +---+------+----------+------+-------+
​               only showing top 5 rows

###   1.2  设置ipython Notebook 字体

```linux
sudo gedit ~/anaconda3/lib/python/site-packages/notebook/static/custom/custom.css
```

```linux
div.output_area pre {
font-family:Courier New;
font-size: 9pt;
}
```

###   1.3 为DataFrame 创建别名

```python
df=user_df.alias("df")
df.show(5)
```

###   1.4 为DataFrame 创建临时表

```python
user_df.createOrReplaceTempView('user_table')
sqlContext.sql("SELECT count(*) counts FROM user_table").show()
# SQL语句很长使用三个引号更方便阅读
sqlContext.sql("""
SELECT count(*) counts
FROM user_table
""").show()
# show()默认显示前20个
sqlContext.sql("""
SELECT * FROM user_table
""").show()
# LIMIT 语句可以减伤加载数据数量，加快加载速度
sqlContext.sql("""
SELECT * FROM user_table LIMIT 5
""").show()
```

###   2.1 SELECT 显示部分字段(4种方式)

```python
# 1.
user_df.select("userid","occupation","gender","age").show(5)
# 2.
user_df.select(user_df.userid,user_df.occupation,user_df.gender,user_df.age).show(5)
# 3.
df.select(df.userid,user_df.occupation,user_df.gender,user_df.age).show(5)
# 4.
df[df['userid'],df['occupation'],df['gender'],df['age']].show(5)
```

###   2.2 增加计算字段

* 使用RDD增加字数字段

  * ```PYTHON
    userNewRDD = userRDD.map(lambda x: (x[0],x[3],x[2],x[1],2016-int(x[1])))
    ```

* 使用DataFrame增加字数字段

  * ```python
    user_df.select("userid","occupation","gender","age", 
                   (2016-df.age).alias('birthyear')).show(5)
    ```

* 使用spark SQL增加字数字段

  * ```PYTHON
    sqlContext.sql("""
    SELECT userid,occupation,gender,age,2016-age bithyear FROM user_table
    """).show(5)
    ```

###   2.3 筛选数据

* 使用RDD筛选数据

  * ```python
    userRDD.filter(lambda r: (r[3]=='technician' and r[2]=='M' and r[1]=='24')).take(5)
    ```

* 使用DataFrame筛选数据

  * ```python
    # 1.
    user_df.filter("occupation='technician'").filter("gender='M'").filter("age=24").show(5)
    # 2.
    df.filter((df.occupation=='technician')&(df.gender=='M')&(df.age==24)).show(5)
    # 3.
    df.filter((df['occupation']=='technician')&(df['gender']=='M')&(df['age']==24)).show(5)
    ```

* 使用spark SQL筛选数据

  * ```python
    sqlContext.sql("""
    SELECT * FROM user_table
    WHERE occupation='technician' and gender='M' and age=24
    """).show(5)
    ```

###   2.4 数据排序

* 使用RDD给数据排序

  * 按单个字段给数据排序

    * ```python
      # 升序
      userRDD.takeOrdered(5, key=lambda x: int(x[1]))
      # 降序
      userRDD.takeOrdered(5, key=lambda x: -1 * int(x[1]))
      ```

  * 按多个字段给数据排序(age降序，gender升序)

    * ```python
      userRDD.takeOrdered(5, key=lambda x: (-int(x[1]),x[2]))
      ```

* 使用DataFrame给数据排序

  * 按单个字段给数据排序

    * ```python
      # 升序
      user_df.select("userid","occupation","gender","age").orderBy("age").show(5)
      df.select("userid","occupation","gender","age").orderBy(df.age).show(5)
      #降序
      user_df.select("userid","occupation","gender","age").orderBy("age",ascending=0).show(5)
      df.select("userid","occupation","gender","age").orderBy(df.age.desc()).show(5)
      ```

  * 按多个字段给数据排序(age降序，gender升序)

    * ```python
      df.select("userid","occupation","gender","age").orderBy(["age","gender"],ascending=[0,1]).show(5)
      df.select("userid","occupation","gender","age").orderBy(df.age.desc(),df.gender).show(5)
      ```

* 使用spark SQL给数据排序

  * 按单个字段给数据排序

    * ```python
      # 升序
      sqlContext.sql("""
      SELECT userid,occupation,gender,age FROM user_table
      ORDER BY age
      """).show(5)
      # 降序
      sqlContext.sql("""
      SELECT userid,occupation,gender,age FROM user_table
      ORDER BY age DESC
      """).show(5)
      ```

  * 按多个字段给数据排序(age降序，gender升序)

    * ```python
      sqlContext.sql("""
      SELECT userid,occupation,gender,age FROM user_table
      ORDER BY age DESC, gender
      """).show(5)
      ```

###   2.5  显示不重复的数据 

* 使用RDD显示不重复的数据

  * ```python
    userRDD.map(lambda x: (x[1],x[2])).distinct().take(5)
    ```

* 使用DataFrame显示不重复的数据

  * ```PYTHON
    df.select("age","gender").distinct().show(5)
    ```

* 使用spark SQL显示不重复的数据

  * ```python
    sqlContext.sql("""
    SELECT distinct age,gender FROM user_table
    """).show(5)
    ```

###   2.6 分组统计数据

* 使用RDD分组统计数据

  * ```python
    userRDD.map(lambda x: (x[2],1)).reduceByKey(lambda x,y: x+y).take(5)
    userRDD.map(lambda x: ((x[2],x[3]),1)).reduceByKey(lambda x,y: x+y).take(5)
    ```

* 使用DataFrame分组统计数据

  * ```python
    df.select(df.gender).groupby(df.gender).count().show(5)
    df.select("gender","occupation").groupBy("gender","occupation").count().orderBy("gender","occupation").show(5)
    df.con
    ```

    * 结果：数据很长，不易阅读

      +----------+----------------------+---------+
      |gender|occupation|count|
      +-----------+---------------------+---------+
      |     F       |administrator  |   36    |
      |     F       |       artist          |   13    |
      |     F       |     educator     |   26    |
      |     F       |     engineer     |    2     |
      |     F       |entertainment|    2     |
      +-----------+---------------------+---------+

  * ```python
    df.stat.crosstab("occupation","gender").show(5)
    ```

    * 结果：这种方式更易阅读

      +----------------------------+--------+-------+
      |occupation_gender|  F       |  M   |
      +----------------------------+--------+-------+
      |        scientist            |    3    |   28  |
      |          student           |   60   |  136 |
      |           writer             |   19   |   26  |
      |         salesman         |    3    |    9   |
      |          retired             |    1    |   13  |
      +----------------------------+---------+-------+

* 使用spark SQL 分组统计数据

  * ```python
    sqlContext.sql("""
    SELECT gender,occupation,count(*) counts 
    FROM user_table
    GROUP BY gender,occupation
    """).show(5)
    ```

###   2.7  join联结

* 下载zipcode-us并导入数据

```python
rawDataWithHeader = sc.textFile(Path + "/data/free-zipcode-database-Primary.csv")
header = rawDataWithHeader.first()
rawData = rawDataHeader.map(lambda x: x!=header)
rData = rawData.map(lambda x: x.replace("\"",""))
zipRDD = rData.map(lambda x: x.split(","))
zipcode_data = zipRDD.map(lambda p:
                         Row(
                         zipcode = int(p[0]),
                         zipcodeType = p[1],
                         city = p[2],
                         state = p[3],
                         ))
```

* 创建DataFrame

```python
zipcode_df = sqlContext.createDataFrame(zipcode_data)
zipcode_df.printSchema()
```

* 创建spark SQL

```python
zipcode_df.createOrReplaceTempView('zipcode_table')
```

* 使用spark SQL进行join联结,并进行查询国家为NY的相关数据

```python
sqlContext.sql("""
SELECT u.*,z.city,z.state
FROM user_table u
LEFT JOIN zipcode_table z ON u.zipcode=z.zipcode
WHERE z.state='NY'
""").show(10)
```

* 根据城市进行统计

```python
sqlContext.sql("""
SELECT z.state,count(*)
FROM user_table u
LEFT JOIN zipcode_table z
WHERE u.zipcode=z.zipcode
GROUP BY z.state
""").show(5)
```

* 对user_df和zipcode_df进行合并为join_df

```python
join_df = user_df.join(zipcode_df,
                      user_df.zipcode==zipcode_df.zipcode,"left_outer")
join_df.printSchema()
join_df.filter("state='NY'").show(5)
```

* 创建GroupByState_df

```python
GroupByState _df= join_df.groupBy("state").count()
```

###   2.8 创建图表

* 以GroupByState_df数据创建图表

  ```python
  import pandas as pd
  import matplotlib.pyplot as plt
  ```

  ```python
  GroupByState_pandas_df = GroupByState.toPandas().set_index("state")
  GroupByState_pandas_df
  
  # 让图表显示在ipython notebook上，否则会单独显示在一个窗口上
  %matplotlib inline
  ax = GroupByState_pandas_df['count'] \
  									.plot(kind='bar',title='state',figsize=(12,6),legend=True,fontsize=12)
  plt.show()
  ```

* 以Occupation_df 数据创建图表

  ```python
  Occupation_df = sqlContext.sql("""
  SELECT u.occupation,count(*) counts
  FROM user_table u
  GROUP BY u.occupation
  """)
  Occupation_pandas_df = Occupation_df.toPandas().set_index("occupation")
  ax = Occupation_pandas_df.plot(kind='pie',
                                title = 'occupation',
                                figsize=(8,8),
                                startangle=90,
                                autopct='%1.1f%%',
                                )
  ax.legend(bbox_anchor(1.05,1),loc=2,borderaxespad=0.)
  plot.show()
  ```

  

## 六. 机器学习流程二元分类(spark ml Pipeline)

###   1.1 使用sqlContext导入数据DataFrame

```python
from pyspark.sql.functions import udf
from pyspark.sql.functions import col
import pyspark.sql.types

global Path
if sc.master[0:5] == "local":
    Path = "file:/usr/local/py_case/pj_SE/"
else:
    Path = "hdfs://master:9000/user/yuanzhss/project"
# 导入DataFrame    
row_df = sqlContext.read.format("csv") \
                .option("header", "true") \
                .option("delimiter","\t") \
                .load(Path + "data/train.tsv")

# 将"?"转换为"0"
def replace_question(x):
    return ('0' if x=='?' else x)
# 使用udf将replace_question(x)转为DataFrame的自定义函数
replace_question = udf(replace_question)
# 1.col(column)读取字段数据
# 2..cast("double")转换为double类型
df = row_df.select(['url','alchemy_category'] + 
                   [replace_question(col(column)).cast("double").alias(column)
                   for column in row_df.columns[4:]])
# 把数据分为train_df,test_df
train_df,test_df = df.randomSplit([0.7,0.3])
train_df.cache()
test_df.cache()
```

###    1.2 机器学习pipeline流程的组件

#### a. StringIndexer

* 可用于将字符串分类特征字段转换为数字

```python
from pyspark.ml.feature import StringIndexer

stringIndexer = StringIndexer(inputCol='alchemy_category',
                              outputCol = 'alchemy_category_index')
categoryTransformer = stringIndexer.fit(df)
df1 = categoryTransformer.transform(train_df)
```

* StringIndexer和.fit 方法会使categoryTransformer生成一个labels的属性

  * ```python
    for i in range(0,len(categoryTransformer.labels)):
        print(str(i) + ':' + categoryTransformer.labels[i])
    df1.select('alchemy_category','alchemy_category_index').show(14)
    ```

  * 结果:(categoryTransformer.labels其实是一个网页分类的字典)

    0:?
    1:recreation
    2:arts_entertainment
    3:business
    4:health
    5:sports
    6:culture_politics
    7:computer_internet
    8:science_technology
    9:gaming
    10:religion
    11:law_crime
    12:unknown
    13:weather

#### b.OneHotEncoder

* 可以将一个数值的分类特征字段转换为多个字段的Vector

```python
from pyspark.ml.feature import OneHotEncoder

encoder = OneHotEncoder(dropLast=False,
                       inputCol='alchemy_category_index',
                       outputCol='alchemy_category_IndexVec')

encoderTransformer = encoder.fit(df1)
df2 = encoderTransformer.transform(df1)
```

* ```python
  df2.select('alchemy_category','alchemy_category_index','alchemy_category_IndexVec').show(5)
  ```

* 结果:

  +-------------------------------+---------------------- ----------------+-----------------------------------------+
  |    alchemy_category   |    alchemy_category_index |alchemy_category_IndexVec|
  +-------------------------------+----------------------------------------+----------------------------------------+
  |          recreation           |                     1.0                       |                (14,[1],[1.0])             | 
  |                    ?                  |                     0.0                       |               (14,[0],[1.0])              |
  |                    ?                  |                     0.0                       |          	 (14,[0],[1.0])		 	 |
  |                    ?                  |                     0.0                       |         	  (14,[0],[1.0])			  |
  |                    ?                  |                     0.0                       |           	(14,[0],[1.0])              |
  +--------------------------------+----------------------------------------+----------------------------------------+

#### c.VectorAssembler

* 可以将多个特征字段整合成一个特征的Vector

```python
from pyspark.ml.feature import VectorAssembler

assemblerInputs = ['alchemy_category_IndexVec'] + row_df.columns[4:-1]
print(assemblerInputs)
assembler = VectorAssembler(
                        inputCols=assemblerInputs,
                        outputCol="features")
df3=assembler.transform(df2)
```

* ```python
  df3.select("features").take(1)
  ```

* 结果：

  [Row(features=SparseVector(36, {1: 1.0, 14: 0.3034, 15: 2.3529, 16: 0.7228, 17: 0.375, 18: 0.3043, 19: 0.288, 20: 0.483, 23: 0.034, 25: 0.223, 26: 0.2186, 27: 1.0, 28: 1.0, 29: 14.0, 31: 9935.0, 32: 184.0, 33: 3.0, 34: 0.3478, 35: 0.1383}))]

#### d.DecisionTreeClassifier

```python
from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(labelCol="label",featuresCol="features",
                           impurity="gini",maxDepth=10,maxBins=14)
dt_model = dt.fit(df3)
df4=dt_model.transform(df3)
```

#### 简洁流程(上述流程方便我们理解)

```python
stringIndexer = StringIndexer(inputCol='alchemy_category',
                              outputCol = 'alchemy_category_index')
encoder = OneHotEncoder(dropLast=False,
                       inputCol='alchemy_category_index',
                       outputCol='alchemy_category_IndexVec')
assemblerInputs = ['alchemy_category_IndexVec'] + row_df.columns[4:-1]
assembler = VectorAssembler(
                        inputCols=assemblerInputs,
                        outputCol="features")
dt = DecisionTreeClassifier(labelCol="label",featuresCol="features",
                           impurity="gini",maxDepth=10,maxBins=14)
pipeline = Pipeline(stages=[stringIndexer,encoder,assembler,dt])

# 使用.fit进行训练
pipelineModel = pipeline.fit(train_df)
# pipelineModel第三阶段会产生决策树模型，可以用下面指令查看模型
pipelineModel.stage[3]
# 查看模型的决策树规则
pipelineModel.stage[3].toDegbugString
# 对test_df数据进行预测
predict = pipelineModel.transform(test_df)
```

* ```python
  print(predict.columns)
  ```

* ['url', 'alchemy_category', 'alchemy_category_score', 'avglinksize', 'commonlinkratio_1', 'commonlinkratio_2', 'commonlinkratio_3', 'commonlinkratio_4', 'compression_ratio', 'embed_ratio', 'framebased', 'frameTagRatio', 'hasDomainLink', 'html_ratio', 'image_ratio', 'is_news', 'lengthyLinkDomain', 'linkwordscore', 'news_front_page', 'non_markup_alphanum_characters', 'numberOfLinks', 'numwords_in_url', 'parametrizedLinkRatio', 'spelling_errors_ratio', 'label', 'alchemy_category_index', 'alchemy_category_IndexVec', 'features', **'rawPrediction', 'probability', 'prediction'**]

```python
# 继续评估模型准确率
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(
                                    rawPredictionCol='rawPrediction',
                                    labelCol="label",
                                    metricName="areaUnderROC")
```

* 使用TrainValidation进行训练验证找出最佳模型

```python
from pyspark.ml.tuning import ParamGridBuilder,TrainValidationSplit

paramGrid = ParamGridBuilder() \
.addGrid(dt.impurity,["gini","entropy"]) \
.addGrid(dt.maxDepth,[5,10,15]) \
.addGrid(dt.maxBins,[10,15,20]).build()

tvs = TrainValidationSplit(estimator=dt, evaluator=evaluator,
                          estimatorParamMaps=paramGrid,trainRatio=0.8)
# 与前面相比，有tvs(增加了参数的模型)代替dt
tvs_pipeline = Pipeline(stages=[stringIndexer,encoder,assembler,tvs])
# 对新模型使用.fit进行训练
tvs_pipelineModel = tvs_pipeline.fit(train_df)
# 找出最佳模型
bestModel = tvs_pipelineModel.stages[3].bestModel
bestModel
# 结果
predictions = tvs_pipelineModel.transform(test_df)
auc = evaluator.evaluate(predictions)
auc
```

###   1.3 使用crossValidation交叉检验找出最佳模型

* k值越大，效果越好

```python
from pyspark.ml.tuning import CrossValidator

cv = CrossValidator(estimator=dt, evaluator=evaluator, estimatorParamMaps = paramGrid, numFolds=2)
cv_pipeline = Pipeline(stages=[stringIndexer,encoder,assembler,cv])
cv_pipelineModel = cv_pipeline.fit(train_df)
cv_predictions = cv_pipelineModel.transform(test_df)
auc = evaluator.evaluate(cv_predictions)

```

###   1.4 使用随机森林RandomForestClassifier分类器

```python
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="label",
                           featuresCol="features",
                           numTrees=10)
rf_pipeline = Pipeline(stages=[stringIndexer,encoder,assembler,rf])
rf_pipelineModel = rf_pipeline.fit(train_df)
rf_predictions = rf_pipelineModel.transform(test_df)
evaluator.evaluate(rf_predictions)
```

###   1.5 完整程序(提高准确率)

* 同时使用crossValidation和RandonForestClassifier

```python
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder,TrainValidationSplit
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.classification import RandomForestClassifier

stringIndexer = StringIndexer(inputCol='alchemy_category',
                              outputCol = 'alchemy_category_index')
encoder = OneHotEncoder(dropLast=False,
                       inputCol='alchemy_category_index',
                       outputCol='alchemy_category_IndexVec')
assemblerInputs = ['alchemy_category_IndexVec'] + row_df.columns[4:-1]
assembler = VectorAssembler(
                        inputCols=assemblerInputs,
                        outputCol="features")
rf = RandomForestClassifier(labelCol="label",
                           featuresCol="features",
                           numTrees=10)

paramGrid = ParamGridBuilder() \
.addGrid(dt.impurity,["gini","entropy"]) \
.addGrid(dt.maxDepth,[5,10,15]) \
.addGrid(dt.maxBins,[10,15,20]) \
.addGrid(rf.numTrees,[10,20,30]).build()

# 使用RandomTree 
rftvs = TrainValidationSplit(estimator=rf,evaluator=evaluator,
                             estimatorParamMaps=paramGrid,trainRatio=0.8)
rftvs_pipeline = Pipeline(stages=[stringIndexer,encoder,assembler,rftvs])
rftvs_pipelineModel = rftvs_pipeline.fit(train_df)
rftvs_predictions = rftvs_pipelineModel.transform(test_df)
auc = evaluator.evaluate(rftvs_predictions)

# 使用CrossValidation
rfcv = CrossValidator(estimator=rf,evaluator=evaluator,
                             estimatorParamMaps=paramGrid,numFolds=2)
rfcv_pipeline = Pipeline(stages=[stringIndexer,encoder,assembler,rfcv])
rfcv_pipelineModel = rfcv_pipeline.fit(train_df)
rfcv_predictions = rfcv_pipelineModel.transform(test_df)
auc = evaluator.evaluate(rfcv_predictions)
auc
```

* 思路分析
  * 把特征类字段转化为数值类字段(StringIndexer Encoder)
  * 把转化后的特征类字段和其他数值类字段合并在一起(vectorAssembler)
  * 使用DecisionTree或RandomTree创建模型(通过features和label)
  * 添加模型创建时使用的参数ParamGrid
    * 如果使用RandomTree还需要添加参数numTrees
  * 通过机器学习pipeline添加各组件到流程中创建模型
  * 通过.fix方法添加数据train_df训练模型
  * 对pipeline模型使用.trainsform方法预测结果test_df
  * 改变使用CrossValidator替换TrainValidationSplit方法验证准确度
    * CrossValidation会比TrainValidationSplit方法更准确
      * 因为TrainValidationSplit只是简单的划分trainData和testData
      * CrossValidator在划分trainData和testData后仍会交叉变更为testData和trainData从而找出最佳模型

## 七. 机器学习流程多元分类

###   1.1 数据准备

```python
global Path
if sc.master[0:5] == "local":
    Path = "file:/usr/local/py_case/pj_UCI/"
else:
    Path = "hdfs://master:9000/user/yuanzhss/project"
    
rawData = sc.textFile(Path + "data/covtype.data")
rawData = rawData.take(5000)
rawData = sc.parallelize(rawData)
lines = rawData.map(lambda x: x.split(","))
lines.count()
```

```python
from pyspark.sql.types import StringType,StructField,StructType
# fieldnum = 55
fieldnum = len(lines.first())
fields = [StructField("f" + str(i), StringType(), True) for i in range(fieldnum)]
schema = StructType(fields)
covtype_df =spark.createDataFrame(lines,schema)
print(covtype_df.columns)
#['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48', 'f49', 'f50', 'f51', 'f52', 'f53', 'f54']

from pyspark.sql.functions import col
covtype_df = covtype_df.select([col(column).cast("double").alias(column) for column in covtype_df.columns])
# 提取features字段
featureCols = covtype_df.columns[:54]
# 把"f54"变为"label",并丢弃f54的标签
covtype_df = covtype_df.withColumn("label",covtype_df["f54"] -1).drop("f54")
```

###   1.2 完整程序(与上述案例大致一样)

```python
# 分为train_df和test_df
train_df,test_df = covtype_df.randomSplit([0.7,0.3])
train_df.cache()
test_df.cache()

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

# 这个案例没有特征类字段，所以只需把所有数值型字段合并即可
vectorAssembler = VectorAssembler(inputCols=featureCols,
                                 outputCol="features")
dt = DecisionTreeClassifier(labelCol="label",
                           featuresCol="features",
                           maxDepth=5,maxBins=20)
dt_pipeline = Pipeline(stages=[vectorAssembler,dt])

pipelineModel = dt_pipeline.fit(train_df)
dt_predicted = pipelineModel.transform(test_df)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# 评估模型准确度
evaluator = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction",
                                             metricName="accuracy")
accuracy = evaluator.evaluate(dt_predicted)

# 使用TrainValidation找出最佳模型
from pyspark.ml.tuning import TrainValidationSplit,ParamGridBuilder

paramGrid = ParamGridBuilder() \
.addGrid(dt.impurity,["gini","entropy"]) \
.addGrid(dt.maxDepth,[10,15,25]) \
.addGrid(dt.maxBins,[30,40,50]) \
.build()
tvs = TrainValidationSplit(estimator=dt, evaluator=evaluator,estimatorParamMaps=paramGrid,trainRatio=0.8)
tvs_pipeline = Pipeline(stages=[vectorAssembler, tvs])
tvs_pipelineModel = tvs_pipeline.fit(train_df)
bestModel = tvs_pipelineModel.stages[1].bestModel
bestModel
# DecisionTreeClassificationModel (uid=DecisionTreeClassifier_03b81f51d327) of depth 20 with 947 nodes
tvs_predictions = tvs_pipelineModel.transform(test_df)
accuracy = evaluator.evaluate(tvs_predictions)
accuracy
```

## 八. 机器学习流程回归

###   1.1 准备数据

```python
global Path
if sc.master[0:5] == "local":
    Path = "file:/usr/local/py_case/pj_UCI_BikeSharing/"
else:
    Path = "hdfs://master:9000/user/yuanzhss/project"
    
hour_df = spark.read.format('csv') \
                    .option("header","true").load(Path + "data/hour.csv")
print(hour_df.columns)
# ['instant', 'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']

# 去除不需要的字段
hour_df = hour_df.drop("instant").drop("dteday") \
                .drop("yr").drop("casual").drop("registered")
# 把字段下的数据转化为"double"类型
from pyspark.sql.functions import col
hour_df = hour_df.select([col(column).cast("double").alias(column) for column in hour_df.columns])

train_df,test_df = hour_df.randomSplit([0.7,0.3])
train_df.cache()
test_df.cache()
```

###   2.1 合并字段

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler,VectorIndexer

featuresCols = hour_df.columns[:-1]
print(featuresCols)
vectorAssembler = VectorAssembler(inputCols=featuresCols,
                                  outputCol="aFeatures")
vectorIndexer = VectorIndexer(inputCol="aFeatures",
                             outputCol="features",maxCategories=24)
```

###   3.1 使用GBT Regression创建模型(也可以使用DecisionTreeRegressor)

```python
from pyspark.ml.regression import GBTRegressor

GBT = GBTRegressor(labelCol="cnt",featuresCol="features")
gbt_pipeline = Pipeline(stages=[vectorAssembler,vectorIndexer,GBT])
gbt_pipelineModel = gbt_pipeline.fit(train_df)
# 检验准确率
gbt_predictions = gbt_pipelineModel.transform(test_df)
rmse = evaluator.evaluate(gbt_predictions)
rmse
```

###   4.1 使用CrossValidation找出最佳模型(也可以使用TrainValidation)

```python
from pyspark.ml.tuning import CrossValidator
cvGBT = CrossValidator(estimator=GBT,evaluator=evaluator,estimatorParamMaps=paramGrid,numFolds=2)
cvGBT_pipeline = Pipeline(stages=[vectorAssembler,vectorIndexer,cvGBT])
cvGBT_pipelineModel = cv_pipeline.fit(train_df)
cvGBT_predictions = cv_pipelineModel.transform(test_df)
rmse = evaluator.evaluate(cvGBT_predictions)
rmse
```



