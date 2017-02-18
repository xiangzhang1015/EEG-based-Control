# EEG-based-Control
This repository is the open Python code and EEG data for the submitted KDD paper "Enhancing Mind Controlled Smart Living Through Recurrent
Neural Networks".
"Data" contains 29738 rows, with per row is one EEG sample has 65 elements. The first 64 elements are the 64 channels EEG raw data collected by BCI2000 system, and the last element is the label of the sample in this row.  The EEG database comes from an open database eegmmidb, see the details here:http://www.physionet.org/pn4/eegmmidb/.

Three files contained "eeg_rawdata_runn_model00.00400.005643.ckpt" are the model I trained through the EEG_based_control_train.py. You can train yourself model. This model is used in EEG_based_control_test.py.

This is a little simple and more comments in the code will be added later.
