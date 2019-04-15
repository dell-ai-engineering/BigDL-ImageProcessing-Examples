import os
import random
import pprint
import io
import json

from flask import Flask, jsonify, request, render_template, send_from_directory, Markup, Response
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from pyspark.sql.types import StringType, ArrayType

from zoo.common.nncontext import *
from zoo.feature.image import *
from zoo.pipeline.nnframes import *
from zoo.pipeline.api.net import Net
from zoo.pipeline.api.keras.layers import *

app = Flask(__name__)
app.jinja_env.filters['zip'] = zip
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

sparkConf = create_spark_conf().setAppName("ChestXray_Inference")
sc = init_nncontext(sparkConf)
spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()
sqlContext = SQLContext(sc)

DEBUG = True
TEST_IMAGES_DIR = "./sample_images/"
LABEL_PATH = "./sample_images/Data_Entry_2017.csv"
MODEL_PATH = "./model/model.bigdl"
MODEL_WEIGHTS_PATH = "./model/model.bin"

LABEL_TEXTS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
               "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
               "Pleural_Thickening", "Hernia"]


def list_images(images_dir=TEST_IMAGES_DIR):
    images_list = [
        img for img in os.listdir(images_dir)
        if img.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
    ]
    return images_list


def text_to_label(text):
    label_map = {k: v for v, k in enumerate(LABEL_TEXTS)}
    arr = [0.0] * len(LABEL_TEXTS)
    for l in text.split("|"):
        if l != "No Finding":
            arr[label_map[l]] = 1.0
    return arr


def create_input_df(image_name, images_dir=TEST_IMAGES_DIR, labels_path=LABEL_PATH):
    getLabel = udf(lambda x: text_to_label(x), ArrayType(DoubleType()))
    getName = udf(lambda row: os.path.basename(row[0]), StringType())
    imageDF = NNImageReader.readImages(
        os.path.join(images_dir, image_name), sc, resizeH=256, resizeW=256,
        image_codec=1) \
        .withColumn("Image Index", getName(col('image'))) \
        .withColumnRenamed('Image Index', 'Image_Index')
    labelDF = sqlContext.read.option('timestampFormat', 'yyyy/MM/dd HH:mm:ss ZZ') \
        .load(labels_path, format="csv", sep=",", inferSchema="true", header="true") \
        .select("Image Index", "Finding Labels") \
        .withColumn("label", getLabel(col('Finding Labels'))) \
        .withColumnRenamed('Image Index', 'Image_Index')
    labelDF1 = labelDF.withColumnRenamed('Image Index', 'Image_Index') \
        .withColumnRenamed('Finding Labels', 'Finding_Labels')
    inputDF = imageDF.join(labelDF1, on="Image_Index", how="inner")
    return inputDF


def load_model(model_path=MODEL_PATH, model_weights_path=MODEL_WEIGHTS_PATH):
    model = Net.load(model_path, model_weights_path)
    return model


def predict(model, inputdf, image_feature_col="image", batchsize=4, debug=DEBUG):
    """
    Predict output of when inputdf is passed through model
    """
    transformer = ChainedPreprocessing([
        RowToImageFeature(),
        ImageCenterCrop(224, 224),
        ImageChannelNormalize(123.68, 116.779, 103.939),
        ImageMatToTensor(),
        ImageFeatureToTensor()])
    classifier_model = NNModel(model, transformer).setFeaturesCol(image_feature_col) \
        .setBatchSize(batchsize)
    outputdf = classifier_model.transform(inputdf)
    collect_df = outputdf.select("Image_Index", "prediction", "label").collect()[0]
    prediction = collect_df.prediction
    true_finding = collect_df.label
    spark.catalog.clearCache()  # to bypass Java heap space error
    return prediction, true_finding, LABEL_TEXTS


def get_patient_xray_ids(image_name):
    name_parts = image_name.split(".")[0].split("_")
    return 'Patient {} | X-Ray {}'.format(name_parts[1], name_parts[0])


def create_prediction_graph(prediction_json):
    prediction_dict = json.loads(prediction_json)
    pprint.pprint(prediction_dict)
    fig = Figure(figsize=(12,6))
    axis = fig.add_subplot(1, 1, 1)
    axis.bar(x=prediction_dict['labels_text'], height=prediction_dict['true_finding'],
             width=0.2, color="red", alpha=1.0, label="Actual Findings")
    axis.bar(x=prediction_dict['labels_text'], height=prediction_dict['prediction'],
             width=0.5, color="blue", alpha=0.5, label="Predicted Probabilities")
    axis.set_xticklabels(prediction_dict['labels_text'], rotation=45, ha="right")
    axis.legend()
    axis.set_axisbelow(True)
    axis.grid()
    axis.set_xlabel("Disease")
    axis.set_ylabel("Probability")
    axis.set_ylim(0,1)
    axis.set_title(get_patient_xray_ids(prediction_dict['image_name']))
    fig.tight_layout()
    return fig


TEST_IMAGES_LIST = list_images(TEST_IMAGES_DIR)
RESNET_MODEL = load_model(MODEL_PATH, MODEL_WEIGHTS_PATH)


@app.route("/test")
def index():
    spark.catalog.clearCache()  # to bypass Java heap space error
    selected_image = random.choice(TEST_IMAGES_LIST)
    selection_sdf = create_input_df(selected_image)
    prediction, true_finding, labels_text = predict(RESNET_MODEL, selection_sdf)
    spark.catalog.clearCache()  # to bypass Java heap space error
    output = {
        'image_name': selected_image,
        'prediction': prediction,
        'true_finding': true_finding,
        'labels_text': labels_text,
    }
    return jsonify(output)


@app.route('/display')
def display_image():
    filename = request.args.get('filename', None)
    return send_from_directory(TEST_IMAGES_DIR, filename)


@app.route('/static')
def display_static():
    filename = request.args.get('filename', None)
    return send_from_directory("./", filename)


@app.route('/prediction')
def predict_finding():
    spark.catalog.clearCache()  # to bypass Java heap space error
    selected_image = request.args.get('filename', None)
    patient_xray_id = get_patient_xray_ids(selected_image)
    selection_sdf = create_input_df(selected_image)
    prediction, true_finding, labels_text = predict(RESNET_MODEL, selection_sdf)
    spark.catalog.clearCache()  # to bypass Java heap space error
    prediction_details = pd.DataFrame({'prediction': prediction, 'true_labels': true_finding},
                                      index=labels_text)
    prediction_details['true_labels'] = prediction_details['true_labels'].astype(int)
    prediction_details['prediction'] = prediction_details['prediction'].round(3)
    prediction_table = Markup(prediction_details.to_html(justify="center"))
    prediction_dict = {
        'image_name': selected_image,
        'prediction': prediction,
        'true_finding': [int(x) for x in true_finding],
        'labels_text': labels_text,
    }
    prediction_json = json.dumps(prediction_dict)
    return render_template("prediction.html", selected_image=selected_image,
                           prediction_table=prediction_table,
                           patient_xray_id=patient_xray_id, prediction_json=prediction_json)


@app.route('/prediction_graph')
def display_prediction_graph():
    prediction_json = request.args.get('prediction_json', None)
    fig = create_prediction_graph(prediction_json)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    response = Response(output.getvalue(), mimetype='image/png')
    return response


@app.route('/')
def get_gallery():
    spark.catalog.clearCache()  # to bypass Java heap space error
    patient_xray_ids = [get_patient_xray_ids(image_name) for image_name in TEST_IMAGES_LIST]
    return render_template("selection.html", image_names=TEST_IMAGES_LIST,
                           patient_xray_ids=patient_xray_ids)


if __name__ == "__main__":
    app.run(host='172.30.10.93', port=6001)
