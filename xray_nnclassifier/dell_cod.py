import random

from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *
from pyspark.sql.types import DoubleType
from pyspark.sql.types import StringType, ArrayType
from zoo.common.nncontext import *
from zoo.feature.image.imagePreprocessing import *
from zoo.pipeline.api.keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
from zoo.pipeline.api.keras.metrics import AUC
from zoo.pipeline.api.keras.models import *
from zoo.pipeline.api.net import Net
from zoo.pipeline.nnframes import *
from zoo.pipeline.api.keras.objectives import BinaryCrossEntropy
import time


def get_auc_for_kth_class(k, df, label_col="label", prediction_col="prediction"):
    get_Kth = udf(lambda a: a[k], DoubleType())
    extracted_df = df.withColumn("kth_label", get_Kth(col(label_col))) \
        .withColumn("kth_prediction", get_Kth(col(prediction_col))) \
        .select('kth_label', 'kth_prediction')
    # areaUnderROC|areaUnderPR
    roc_score = BinaryClassificationEvaluator(rawPredictionCol='kth_prediction',
                                              labelCol='kth_label',
                                              metricName="areaUnderROC") \
        .evaluate(extracted_df)

    print('roc score for ', k, ' class: ', roc_score)
    return roc_score


def get_inception_model(model_path, label_length):
    full_model = Net.load_bigdl(model_path)
    model = full_model.new_graph(["pool5/drop_7x7_s1"])  # this inception
    inputNode = Input(name="input", shape=(3, 224, 224))
    inception = model.to_keras()(inputNode)
    flatten = Flatten()(inception)
    logits = Dense(label_length, activation="sigmoid")(flatten)
    lrModel = Model(inputNode, logits)
    return lrModel

def get_resnet_model(model_path, label_length):
    full_model = Net.load_bigdl(model_path)
    model = full_model.new_graph(["pool5"])  # this inception
    print(('num of model layers: ', len(model.layers)))
    inputNode = Input(name="input", shape=(3, 224, 224))
    inception = model.to_keras()(inputNode)
    flatten = GlobalAveragePooling2D(dim_ordering='th')(inception)
    logits = Dense(label_length, activation="sigmoid")(flatten)
    lrModel = Model(inputNode, logits)
    return lrModel

if __name__ == "__main__":
    random.seed(1234)

    batch_size = int(sys.argv[1])
    num_epoch = int(sys.argv[2])

    model_path = sys.argv[3] #"/home/yuhao/workspace/model/bigdl_inception-v1_imagenet_0.4.0.model"
    image_path = sys.argv[4] #"/home/yuhao/workspace/data/xray/middle_images"
    save_path = sys.argv[5] #"./save_model"
    label_length = 14

    sparkConf = create_spark_conf().setAppName("test_dell_x_ray")
    sc = init_nncontext(sparkConf)
    spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()
    print(sc.master)

    xray_model = get_resnet_model(model_path, label_length)

    sliceLabel = udf(lambda x: x[:label_length], ArrayType(DoubleType()))
    train_df = spark.read.load(image_path).withColumn("part_label", sliceLabel(col('label')))
    (trainingDF, validationDF) = train_df.randomSplit([0.7, 0.3])
    trainingDF.cache()
    validationDF.cache()
    print("training df count: ", trainingDF.count())

    #logdir ='/logDirectory'
    # train_summary = TrainSummary(log_dir="./logs", app_name="testNNClassifer")
    # val_summary = ValidationSummary(log_dir="./logs", app_name="testNNClassifer")
    # train_summary.set_summary_trigger("Parameters", SeveralIteration(1))
    # train_summary.set_summary_trigger("LearningRate", SeveralIteration(1))

    transformer = ChainedPreprocessing(
        [RowToImageFeature(), ImageCenterCrop(224, 224),
         ImageChannelNormalize(123.68, 116.779, 103.939), ImageMatToTensor(), ImageFeatureToTensor()])

    classifier = NNEstimator(xray_model, BinaryCrossEntropy(), transformer, SeqToTensor([label_length])) \
        .setBatchSize(batch_size)\
        .setMaxEpoch(num_epoch) \
        .setFeaturesCol("image")\
        .setLabelCol("part_label") \
        .setOptimMethod(SGD(learningrate=0.001, leaningrate_schedule=Plateau("Loss", factor=0.1, patience=1, mode="min", epsilon=0.01, cooldown=0, min_lr=1e-15))) \
        .setCachingSample(False)\
        # .setValidation(EveryEpoch(), validationDF, [AUC()], batch_size)\
         # .setTrainSummary(train_summary) \
         # .setValidationSummary(val_summary) \
         # .setCheckpoint("./checkpoints", EveryEpoch(), False)

    start = time.time()
    nnModel = classifier.fit(trainingDF)
    print("Finished training, took: ", time.time() - start)

    model_path = save_path + '/xray_model_' + time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    nnModel.save(model_path)
    print('model saved at: ', model_path)
    predictionDF = nnModel.transform(validationDF).cache()

    for i in range(0, label_length):
        get_auc_for_kth_class(i, predictionDF)

    print("Finished evaluation")

