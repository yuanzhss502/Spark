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

