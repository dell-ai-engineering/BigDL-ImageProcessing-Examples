
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

label_texts = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
     "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No Finding"]
label_map = {k: v for v, k in enumerate(label_texts)}

def text_to_label(text):
    arr = [0.0] * len(label_texts)
    for l in text.split("|"):
        arr[label_map[l]] = 1.0
    return arr

label_length = 14
sliceLabel = udf(lambda x: x[:label_length], ArrayType(DoubleType()))

getLabel = udf(lambda x: text_to_label(x), ArrayType(DoubleType()))
getName = udf(lambda row: os.path.basename(row[0]), StringType())
imageDF = NNImageReader.readImages(image_path, sc, minPartitions=400, resizeH=256, resizeW=256, image_codec=1) \
    .withColumn("Image Index", getName(col('image')))
imageDF=imageDF.withColumnRenamed('Image Index', 'Image_Index')
labelDF = spark.read.load(label_path, format="csv", sep=",", inferSchema="true", header="true") \
    .select("Image Index", "Finding Labels") \
    .withColumn("label", getLabel(col('Finding Labels'))) \
    .withColumn("part_label", sliceLabel(col('label'))) \
    .withColumnRenamed('Image Index', 'Image_Index') \
    .select("Image_Index", "label", "part_label")
labelDF.printSchema()

train_df = imageDF.join(labelDF, on="Image_Index", how="inner")
(trainingDF, validationDF) = train_df.randomSplit([0.7, 0.3], seed=37)

trainingDF.write.save(save_path + "/trainingDF")
validationDF.write.save(save_path + "/validationDF")

print("data saved at ", save_path)