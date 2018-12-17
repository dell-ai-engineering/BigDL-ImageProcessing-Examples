${ANALYTICS_ZOO_HOME}/bin/spark-submit-with-zoo.sh \
--master local[1] \
--driver-memory 10g \
data.py \
/home/yuhao/workspace/data/xray/middle_images /home/yuhao/workspace/github/hhbyyh/BigDL-ImageProcessing-Examples/xray_nnclassifier ./imageDF


#${ANALYTICS_ZOO_HOME}/bin/spark-submit-with-zoo.sh \
#--master yarn \
#--deploy-mode client \
#--executor-memory 150g \
#--driver-memory 150g \
#--executor-cores 10 \
#--num-executors 4 \
#convert_image.py \
#hdfs://Gondolin-Node-071:9000/images/images_*/* hdfs://Gondolin-Node-071:9000/labels/Data_Entry_2017.csv hdfs://Gondolin-Node-071:9000/imageDF3




