############################
# Start a new VM in GCP
############################
1. Testing was done with CentOS 7

############################
# Install pip and tensorflow
############################
% sudo yum install python-pip 
% sudo pip install --upgrade pip
% sudo yum -y install epel-release
% sudo yum -y install gcc gcc-c++ python-pip python-devel atlas atlas-devel gcc-gfortran openssl-devel libffi-devel
% sudo pip install --upgrade numpy scipy wheel cryptography 
% sudo pip install --upgrade tensorflow

############################
# Train and Export Model 
############################
# Download files in Google Cloud Platform
# Usage: mnist_saved_model.py [--training_epochs=x] [--model_version=y] export_dir
% python trainExportMNIST.py --training_epochs=6 --model_version=1 exported_mnist_model

# Test model for sanity
saved_model_cli show --dir exported_mnist_model/1/ --all

# Test model on two images
saved_model_cli run --dir exported_mnist_model/1 --tag_set serve --signature_def predict_images --inputs images=two_images.npy 

############################
# Transfer model to bucket
############################
% gsutil cp -r exported_mnist_model gs://exported_mnist_model
