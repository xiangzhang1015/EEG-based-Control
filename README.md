# EEG-based-Control
This repository is the open Python code and EEG data for the submitted KDD paper "Enhancing Mind Controlled Smart Living Through Recurrent
Neural Networks".

"Data.mat" contains 29738 rows, with per row is one EEG sample has 65 elements. The first 64 elements are the 64 channels EEG raw data collected by BCI2000 system, and the last element is the label of the sample in this row.  The EEG database comes from an open database eegmmidb, see the details here:http://www.physionet.org/pn4/eegmmidb/.

robot_command.mat is the command data for 'Assisted Living with Mind-controlled Mobile Robot' simulation. It controls the robot under ROS system complete the task of grasp a cup of water and bring it to the livingroom's table.

EEG_based_control_train.py and EEG_based_control_test.py separately is the train and test RNN model, some basic comments of the code is contained in the code. Download and try it for your own data. 

Three files contained "eeg_rawdata_runn_model00.00400.005643.ckpt" are the model I trained through the EEG_based_control_train.py. You can train yourself model. This model is used in EEG_based_control_test.py.

This is a little simple and more comments in the code will be added later.

I uploaded the New_EEG_Control.py code which can run on the GPU. It's hundred times faster than on CPU. It only takes 165 seconds for 3500 iterations. 

Another point I found is that if we add sigmoid function in the first hidden layer, the accuracy increase very fast but can only up to 0.9505 as shown in my paper.
However, if we don't add the sigmoid function, the accuracy increase slowly but can achieve 0.973 after around 3500 iterations.
