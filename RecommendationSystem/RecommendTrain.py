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
    
