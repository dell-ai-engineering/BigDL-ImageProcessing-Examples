## Summary
In this notebook, we demonstrate how we can we can build an end to end deep learning pipeline on Spark leveraging the Analytics Zoo for an image processing problem. Distributed Spark worker nodes are used to train our deep learning model at scale. We used the [Chest Xray dataset](http://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a) released by the National Health Institute to develop an AI models to diagnose pneumonia, emphysema, and other thoracic pathologies from chest x-rays. Using the Stanford University [CheXNet](https://stanfordmlgroup.github.io/projects/chexnet/) model as inspiration, we explore ways of developing accurate models for this problem on a distributed Spark cluster. We explore various neural network topologies to gain insight into what types of modelsneural networks scale well in parallel and reduceimprove training time from days to hours. 

Refer to the [white paper](https://www.dellemc.com/resources/en-us/asset/white-papers/solutions/h17686_hornet_wp.pdf) for more information on this study.

## Environment
- Python 2.7 or higher 
- JDK 8 
- Apache Spark 2.1.1 or higher
- Jupyter Notebook 4.1 or spark submit using CLI(Command Line Interface). 
- BigDL 0.7.0 or higher 
- Analytics zoo 0.4.0 or higher

## Hardware Infrastructure
- Hadoop cluster with at least 4 nodes with driver memory 170GB and executor memory is 170GB.

## Download and Install Analytics Zoo and BigDL 
- Download from this [link](https://repo1.maven.org/maven2/com/intel/analytics/bigdl/dist-spark-2.1.1-scala-2.11.8-all/0.7.0/dist-spark-2.1.1-scala-2.11.8-all-0.7.0-dist.zip) to install and configure BigDL and for analytics zoo follow this [link](https://oss.sonatype.org/content/repositories/releases/com/intel/analytics/zoo/analytics-zoo-bigdl_0.7.1-spark_2.1.1/0.3.0/analytics-zoo-bigdl_0.7.1-spark_2.1.1-0.3.0-dist-all.zip). 
- Follow these documentation links for the detailed steps on how to install and configure [BigDL](https://bigdl-project.github.io/0.7.0/#) and [Analytics Zoo](https://analytics-zoo.github.io/0.4.0/index.html).

## Run jupyter Notebook 
- Run export SPARK_HOME = the root directory of Spark. (Ex: /opt/cloudera/parcels/SPARK2-2.1.0.cloudera2-1.cdh5.7.0.p0.171658/lib/spark2)
- Run export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package. (Ex: /usr/lib/zoo)
- Run the following bash command to start the jupyter notebook. Change parameter settings as you need,
```bash
$ANALYTICS_ZOO_HOME/bin/jupyter-with-zoo.sh  \
    --master yarn \
    --num-executors 4 \
    --executor-cores 16 \
    --driver-memory 170g \
    --executor-memory 170g 
```

## Run spark-submit
- Run export SPARK_HOME = the root directory of Spark. (Ex: /opt/cloudera/parcels/SPARK2-2.1.0.cloudera2-1.cdh5.7.0.p0.171658/lib/spark2)
- Run export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package. (Ex: /usr/lib/zoo)
- Run the following spark-submit command. Change parameter settings as you need,
```bash	
$ANALYTICS_ZOO_HOME/bin/spark-submit-with-zoo.sh \
    --master yarn \
    --deploy-mode cluster \
    --num-executors 4 \
    --executor-cores 8 \
    --driver-memory 300g \
    --executor-memory 300g \
    path/to/python_file.py \
    batch_size \
    num_epochs \
    path/to/pretrained model file \
    path/to/dataset \ 
    path/to/save the model
```

