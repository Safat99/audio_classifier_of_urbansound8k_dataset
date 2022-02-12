# audio_classifier_of_urbansound8k_dataset
the repo consists of a gui based audio classifier based on CNN<br>
To run the project one has to do the following things:
## steps:
* download the *features_files_obtained_from_the_dataset* .h5 files from the [link](https://drive.google.com/file/d/1Y7O0bwBspfwUp42_42JPiWqH8IeSp3av/view?usp=sharing)
* download the training models from [here](https://drive.google.com/file/d/1UwS3YY1hAjbx5f6pAS1QmD4w4TgX25wE/view?usp=sharing)
* keep all files in the same directory
* make a virtual envrionment by  ``` python3 -m virtualenv env ```
* activate the virtual environment by ``` source env/bin/activate ```
* install all the required python packages by ``` pip install -r requirements.txt ```
* run the main application or GUI based simple classifier by ``` python combined_for_detection.py ```
* load the sample audio(preferred .wav files of short 4 sec clips, as the audio files of the dataset are .wav file) and see the result in shell

### demo of the GUI
![](https://github.com/Safat99/audio_classifier_of_urbansound8k_dataset/blob/main/GUI%20screenshot.png)
