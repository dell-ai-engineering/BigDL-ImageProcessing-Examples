
from bigdl.nn.criterion import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *
from pyspark.sql.types import DoubleType
from pyspark.sql.types import StringType, ArrayType
from zoo.common.nncontext import *
from zoo.pipeline.nnframes import *

image_path = sys.argv[1] #"/home/yuhao/workspace/data/xray/middle_images"
label_path = sys.argv[2] #"/home/yuhao/workspace/data/xray/Data_Entry_2017.csv"
save_path = sys.argv[3] #"./save_model"

sparkConf = create_spark_conf().setAppName("test_dell_x_ray")
sc = init_nncontext(sparkConf)
spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()
print(sc.master)

label_texts = list("""Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural_Thickening, Cardiomegaly, Nodule, Mass, Hernia, No Finding""".replace("\n", "").split(", "))
label_map = {k: v for v, k in enumerate(label_texts)}

def text_to_label(text):
    arr = [0.0] * len(label_texts)
    for l in text.split("|"):
        arr[label_map[l]] = 1.0
    return arr

getLabel = udf(lambda x: text_to_label(x), ArrayType(DoubleType()))
getName = udf(lambda row: os.path.basename(row[0]), StringType())
imageDF = NNImageReader.readImages(image_path, sc, minPartitions=400, resizeH=256, resizeW=256, image_codec=1) \
    .withColumn("Image Index", getName(col('image')))
imageDF=imageDF.withColumnRenamed('Image Index', 'Image_Index')
labelDF = spark.read.load(label_path, format="csv", sep=",", inferSchema="true", header="true") \
   .select("Image Index", "Finding Labels") \
   .withColumn("label", getLabel(col('Finding Labels')))

labelDF = labelDF.withColumnRenamed('Image Index', 'Image_Index')\
    .withColumnRenamed('Finding Labels', 'Finding_Labels')
labelDF.printSchema()

train_df = imageDF.join(labelDF, on="Image_Index", how="inner")

train_df.write.save(save_path)

spark.read.load(save_path).show()



#label_texts = list("""Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural_Thickening, Cardiomegaly, Nodule, Mass, Hernia, No Finding""".replace("\n", "").split(", "))
#label_map = {k: v for v, k in enumerate(label_texts)}


#def text_to_label(text):
#    arr = [0.0] * len(label_texts)
#    for l in text.split("|"):
#        arr[label_map[l]] = 1.0
#    return arr

#getLabel = udf(lambda x: text_to_label(x), ArrayType(DoubleType()))
#getName = udf(lambda row: os.path.basename(row[0]), StringType())
#imageDF = NNImageReader.readImages(image_path, sc, resizeH=256, resizeW=256, image_codec=1)
#imageDF.show()
#print('image df count: ', imageDF.count())
#imageDF = imageDF.withColumn("Image Index", getName(col('image')))
#imageDF=imageDF.withColumnRenamed('Image Index', 'Image_Index')
#labelDF = spark.read.load(label_path, format="csv", sep=",", inferSchema="true", header="true") \
#   .select("Image Index", "Finding Labels") \
#   .withColumn("label", getLabel(col('Finding Labels')))

#labelDF = labelDF.withColumnRenamed('Image Index', 'Image_Index')\
#    .withColumnRenamed('Finding Labels', 'Finding_Labels')
#labelDF.printSchema()
#print("label count:", labelDF.count())
#print('image df count: ', imageDF.count())

#train_df = imageDF.join(labelDF, on="Image_Index", how="inner")
