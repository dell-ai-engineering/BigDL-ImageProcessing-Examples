${ANALYTICS_ZOO_HOME}/bin/spark-submit-with-zoo.sh \
--master yarn \
--deploy-mode client \
--executor-memory 150g \
--driver-memory 150g \
--executor-cores 10 \
--num-executors 4 \
convert_image.py hdfs://Gondolin-Node-071:9000/images/images_*/* hdfs://Gondolin-Node-071:9000/labels/Data_Entry_2017.csv hdfs://Gondolin-Node-071:9000/imageDF3
