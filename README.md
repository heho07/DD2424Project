# DD2424 Deep Learning Project

###Folder structure
Code folder saves all the code
ResultsFromtraining folder saves .csv documents with training & validationa accuracy during the training run
WeightsFromTraining folder saves the weights


## Up and running

[Tensorflow](https://www.tensorflow.org/) is required to run this project.

### Docker

If you have [Docker](https://docs.docker.com/install/) installed you can pull the tensorflow image by:

    docker pull tensorflow/tensorflow:latest-py3 # CPU version

And run the container:

    ./start-tensorflow.sh
    cd work/ # you will find project files under directory /work.

For details you can read [tensorflow docker](https://www.tensorflow.org/install/docker) or [MIT 6.S094 Settting Up Docker and TensorFlow](https://selfdrivingcars.mit.edu/setting-up-docker-and-tensorflow-for-mac-os-x/)
