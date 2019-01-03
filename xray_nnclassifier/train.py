import random
import time

from math import ceil
from bigdl.optim.optimizer import SGD, SequentialSchedule, Warmup, Poly, Plateau, EveryEpoch, TrainSummary,\
    ValidationSummary, SeveralIteration, Step
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from pyspark.storagelevel import StorageLevel
from zoo.common.nncontext import *
from zoo.feature.image.imagePreprocessing import *
from zoo.feature.common import ChainedPreprocessing
from zoo.pipeline.api.keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D, Dropout
from zoo.pipeline.api.keras.metrics import AUC
from zoo.pipeline.api.keras.optimizers import Adam
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.net import Net
from zoo.pipeline.nnframes import NNEstimator
from zoo.pipeline.api.keras.objectives import BinaryCrossEntropy


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
    dropout = Dropout(0.1)(flatten)
    logits = Dense(label_length, activation="sigmoid")(dropout)
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


def get_sgd_optimMethod(num_epoch, trainingCount, batchSize):
    iterationPerEpoch = int(ceil(float(trainingCount) / batchSize))
    # maxIteration = num_epoch * iterationPerEpoch
    warmupEpoch = 10
    warmup_iteration = warmupEpoch * iterationPerEpoch
    init_lr = 1e-6
    maxlr = 0.001 * batch_size / 8
    print("peak lr is: ", maxlr)
    warmupDelta = (maxlr - init_lr) / warmup_iteration
    cooldownIteration = (num_epoch - warmupEpoch) * iterationPerEpoch

    lrSchedule = SequentialSchedule(iterationPerEpoch)
    lrSchedule.add(Warmup(warmupDelta), warmup_iteration)
    #lrSchedule.add(Step(iterationPerEpoch * 10, 0.1), cooldownIteration)
    lrSchedule.add(Plateau("Loss", factor=0.1, patience=1, mode="min", epsilon=0.01, cooldown=0, min_lr=1e-15), cooldownIteration)
    optim = SGD(learningrate=init_lr, momentum=0.9, dampening=0.0, nesterov=True,
                leaningrate_schedule=lrSchedule)
    return optim


def get_adam_optimMethod(num_epoch, trainingCount, batchSize):
    iterationPerEpoch = int(ceil(float(trainingCount) / batchSize))
    warmupEpoch = 5
    warmup_iteration = warmupEpoch * iterationPerEpoch
    init_lr = 1e-7
    maxlr = 0.001
    print("peak lr is: ", maxlr)
    warmupDelta = (maxlr - init_lr) / warmup_iteration
    cooldownIteration = (num_epoch - warmupEpoch) * iterationPerEpoch

    lrSchedule = SequentialSchedule(iterationPerEpoch)
    lrSchedule.add(Warmup(warmupDelta), warmup_iteration)
    lrSchedule.add(Plateau("Loss", factor=0.1, patience=1, mode="min", epsilon=0.01, cooldown=0, min_lr=1e-15),
                   cooldownIteration)
    optim = Adam(lr=init_lr, schedule=lrSchedule)
    return optim


def evaluate(testDF):
    predictionDF = nnModel.transform(testDF).persist(storageLevel=StorageLevel.DISK_ONLY)
    label_texts = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
                   "Pneumothorax", "Consolidation",
                   "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
    total_auc = 0.0
    for i in range(label_length):
        roc_score = get_auc_for_kth_class(i, predictionDF)
        total_auc += roc_score
        print('{:>12} {:>25} {:>5} {:<20}'.format('roc score for ', label_texts[i], ' is: ', roc_score))

    print("Finished evaluation, average auc: ", total_auc / float(label_length))


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


if __name__ == "__main__":
    random.seed(1234)

    batch_size = int(sys.argv[1])
    num_epoch = int(sys.argv[2])

    model_path = sys.argv[3] #"/home/yuhao/workspace/model/bigdl_inception-v1_imagenet_0.4.0.model"
    data_path = sys.argv[4] #"/home/yuhao/workspace/data/xray/middle_images"
    save_path = sys.argv[5] #"./save_model"
    label_length = 14

    sparkConf = create_spark_conf().setAppName("test_dell_x_ray")
    sc = init_nncontext(sparkConf)
    spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()
    print(sc.master)

    xray_model = get_resnet_model(model_path, label_length)
    train_df = spark.read.load(data_path + '/trainingDF')
    split_id = '00000040_000.png'  #'00024779_000.png'
    trainingDF = train_df.filter(train_df["Image_Index"] < split_id)
    validationDF = train_df.filter(train_df["Image_Index"] >= split_id)
    trainingCount = trainingDF.count()
    print("num of training images: ", trainingCount)
    print("num of validation images: ", validationDF.count())

    transformer = ChainedPreprocessing(
        [RowToImageFeature(), ImageCenterCrop(224, 224), ImageRandomPreprocessing(ImageHFlip(), 0.5),
         ImageChannelNormalize(123.68, 116.779, 103.939), ImageMatToTensor(), ImageFeatureToTensor()])

    train_summary = TrainSummary(log_dir="./logs", app_name="test_dell_x_ray")
    val_summary = ValidationSummary(log_dir="./logs", app_name="test_dell_x_ray")
    train_summary.set_summary_trigger("LearningRate", SeveralIteration(1))
    train_summary.set_summary_trigger("Loss", SeveralIteration(1))

    optim_method = get_adam_optimMethod(num_epoch, trainingCount, batch_size)
    # get_sgd_optimMethod(num_epoch, trainingCount, batch_size)
    classifier = NNEstimator(xray_model, BinaryCrossEntropy(), transformer) \
        .setBatchSize(batch_size) \
        .setMaxEpoch(num_epoch) \
        .setFeaturesCol("image") \
        .setCachingSample(False) \
        .setValidation(EveryEpoch(), validationDF, [AUC()], batch_size) \
        .setTrainSummary(train_summary) \
        .setValidationSummary(val_summary) \
        .setOptimMethod(optim_method)

        # .setEndWhen(MaxIteration(1))
        # .setCheckpoint("./checkpoints", EveryEpoch(), False)

    start = time.time()
    nnModel = classifier.fit(trainingDF)
    print("Finished training, took: ", time.time() - start)

    model_path = save_path + '/xray_model_' + time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    nnModel.model.saveModel(model_path + ".bigdl", model_path + ".bin", True)
    print('model saved at: ', model_path)

    print("evaluating on training data: ")
    evaluate(trainingDF)
    SQLContext(sc).clearCache()

    print("\nevaluating on test data: ")
    testDF = spark.read.load(data_path + '/testDF')
    evaluate(testDF)


    # nnModel.model.to_model().saveModel("/home/yuhao/workspace/github/hhbyyh/BigDL-ImageProcessing-Examples/xray_nnclassifier/kerasModel/m.bigdl",
    #                         "/home/yuhao/workspace/github/hhbyyh/BigDL-ImageProcessing-Examples/xray_nnclassifier/kerasModel/m.bin", True)
    # kk = BigDLModel.loadModel("/home/yuhao/workspace/github/hhbyyh/BigDL-ImageProcessing-Examples/xray_nnclassifier/kerasModel/m.bigdl",
    #                         "/home/yuhao/workspace/github/hhbyyh/BigDL-ImageProcessing-Examples/xray_nnclassifier/kerasModel/m.bin")

    #(trainingDF, validationDF) = train_df.randomSplit([0.7, 0.3])
    #validationDF.cache()
    #logdir ='/logDirectory'
    # train_summary = TrainSummary(log_dir="./logs", app_name="testNNClassifer")
    # val_summary = ValidationSummary(log_dir="./logs", app_name="testNNClassifer")
    # train_summary.set_summary_trigger("Parameters", SeveralIteration(1))
    # train_summary.set_summary_trigger("LearningRate", SeveralIteration(1))

