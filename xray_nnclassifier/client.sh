PYTHONPATH=${ANALYTICS_ZOO_PY_ZIP}:$PYTHONPATH
VENV_HOME=/root/dist/bin

PYSPARK_DRIVER_PYTHON=${VENV_HOME}/venv/bin/python PYSPARK_PYTHON=venv.zip/venv/bin/python ${ANALYTICS_ZOO_HOME}/bin/spark-submit-with-zoo.sh \
--master yarn \
--deploy-mode client \
--executor-memory 170g \
--driver-memory 170g \
--executor-cores 4 \
--num-executors 4 \
--archives ${VENV_HOME}/venv.zip \
xray_test.py 256 5 analytics-zoo_resnet-50_imagenet_0.1.0.model \
hdfs://Gondolin-Node-071:9000/imageDF3 /root/save_models
