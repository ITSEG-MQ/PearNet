# PearNet

## Prepare datasets
Two public datasets are used in this study:
[Sleep-EDF-20](https://gist.github.com/emadeldeen24/a22691e36759934e53984289a94cb09b),
[Sleep-EDF-78](https://physionet.org/content/sleep-edfx/1.0.0/)

After downloading the datasets, the data can be prepared as follows:
```
`cd prepare_datasets`
python prepare_physionet.py --data_dir /path/to/PSG/files --output_dir edf_20_npz --select_ch "EEG Fpz-Cz"
python prepare_physionet.py --data_dir /path/to/PSG/files --output_dir edf_78_npz --select_ch "EEG Fpz-Cz"
```
## Training PearNet
The `config.json` file is used to update the training parameters.
To perform the standard K-fold crossvalidation, specify the number of folds in `config.json` and run the following:
```
chmod +x batch_train.sh
./batch_train.sh 0 /path/files
```
where the first argument represents the GPU id.

## Results
The log file of each fold is found in the fold directory inside the save_dir.   

