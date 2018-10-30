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
    model = full_model.new_graph(["pool5"])
    print(('num of model layers: ', len(model.layers)))
    inputNode = Input(name="input", shape=(3, 224, 224))
    inception = model.to_keras()(inputNode)
    flatten = GlobalAveragePooling2D(dim_ordering='th')(inception)
    logits = Dense(label_length, activation="sigmoid")(flatten)
    lrModel = Model(inputNode, logits)
    return lrModel


def get_vgg_model(model_path, label_length):
    full_model = Net.load_bigdl(model_path)
    model = full_model.new_graph(["pool5"])
    print(('num of model layers: ', len(model.layers)))
    inputNode = Input(name="input", shape=(3, 224, 224))
    inception = model.to_keras()(inputNode)
    flatten = GlobalAveragePooling2D(dim_ordering='th')(inception)
    logits = Dense(label_length, activation="sigmoid")(flatten)
    lrModel = Model(inputNode, logits)
    return lrModel

def get_densenet_model(model_path, label_length):
    full_model = Net.load_bigdl(model_path)
    model = full_model.new_graph(["pool5"])
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

    xray_model = get_inception_model(model_path, label_length)

    sliceLabel = udf(lambda x: x[:label_length], ArrayType(DoubleType()))
    train_df = spark.read.load(image_path).withColumn("part_label", sliceLabel(col('label')))
    (trainingDF, validationDF) = train_df.randomSplit([0.7, 0.3])
    trainingDF.cache()
    validationDF.cache()
    #logdir ='/logDirectory'
    # train_summary = TrainSummary(log_dir="./logs", app_name="testNNClassifer")
    # val_summary = ValidationSummary(log_dir="./logs", app_name="testNNClassifer")
    # train_summary.set_summary_trigger("Parameters", SeveralIteration(1))
    # train_summary.set_summary_trigger("LearningRate", SeveralIteration(1))

    transformer = ChainedPreprocessing(
        [RowToImageFeature(), ImageCenterCrop(224, 224),
         ImageChannelNormalize(123.68, 116.779, 103.939), ImageMatToTensor(), ImageFeatureToTensor()])

    classifier = NNEstimator(xray_model, BinaryCrossEntropy(), transformer, SeqToTensor([label_length])) \
        .setBatchSize(batch_size) \
        .setMaxEpoch(num_epoch) \
        .setFeaturesCol("image")\
        .setLabelCol("part_label") \
        .setOptimMethod(Adam(learningrate=0.001, learningrate_decay=1e-5)) \
        .setCachingSample(False) \
        # .setEndWhen(MaxIteration(1))
        #\
        # .setOptimMethod(SGD(learningrate=0.001, leaningrate_schedule=Plateau("Loss", factor=0.1, patience=1, mode="min", epsilon=0.01, cooldown=0, min_lr=1e-15))) \
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
    trainingDF.unpersist(True)
    predictionDF = nnModel.transform(validationDF).cache()

    label_texts = list(
        """Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural_Thickening, Cardiomegaly, Nodule, Mass, Hernia, No Finding""".replace(
            "\n", "").split(", "))
    label_map = {k: v for v, k in enumerate(label_texts)}
    chexnet_order = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
     "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

    total_auc = 0.0
    for d in chexnet_order:
        roc_score = get_auc_for_kth_class(label_map[d], predictionDF)
        total_auc += roc_score
        print('{:>12} {:>25} {:>5} {:<20}'.format('roc score for ', d, ' is: ', roc_score))

    print("Finished evaluation, average auc: ", total_auc / float(label_length))

