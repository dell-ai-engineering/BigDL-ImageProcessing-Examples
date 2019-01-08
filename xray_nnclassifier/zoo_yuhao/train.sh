# VENV_HOME=/root/dell/dist/bin
# export SPARK_HOME=/root/spark-2.1.0-bin-hadoop2.7/
# export ANALYTICS_ZOO_HOME=/root/dell/dist
#
# PYSPARK_DRIVER_PYTHON=${VENV_HOME}/venv/bin/python PYSPARK_PYTHON=venv.zip/venv/bin/python nohup ${ANALYTICS_ZOO_HOME}/bin/spark-submit-with-zoo.sh \
# --master yarn \
# --deploy-mode client \
# --executor-memory 170g \
# --driver-memory 170g \
# --executor-cores 8 \
# --num-executors 8 \
# --archives ${VENV_HOME}/venv.zip \
# train.py 1024 20 analytics-zoo_resnet-50_imagenet_0.1.0.model \
# hdfs://Gondolin-Node-058:9000/imageDF1 /root/save_models > log.output 2>&1 &


$ANALYTICS_ZOO_HOME/bin/spark-submit-with-zoo.sh \
--master local[1] \
--driver-memory 15g \
train.py \
8 10 /home/yuhao/workspace/model/analytics-zoo_resnet-50_imagenet_0.1.0.model \
/home/yuhao/workspace/github/hhbyyh/BigDL-ImageProcessing-Examples/xray_nnclassifier/zoo_yuhao/imageDF \
/home/yuhao/workspace/github/hhbyyh/BigDL-ImageProcessing-Examples/xray_nnclassifier/zoo_yuhao/save_model

