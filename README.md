# CS598_Sleep_Stage_Classification

Sleep stage classification is important for sleep disorder diagnosis. Emerging deep learning studies in sleep stage classification have demonstrated state-of-the-art performance and can greatly reduce manual effort from human experts. In this paper we present and evaluate several machine learning models that automatically classify sleep stages. They include convolutional neural network (CNN) model, CNN and recurrent neural network (RNN) model, and selected non-deep learning models. All models are developed without any signal preprocessing or hand-engineered features. The CNN-RNN model achieved the best performance among all models with an accuracy of 0.79, average F1 score of 0.72, and cohen’s kappa coefficient of 0.72. The labels produced by our model are in substantial agreement with that from a human expert.

# Data

Our data was too big to fit into this repo, so it is currently hosted on cloud storage outside of this repo. In order to get started with this project you should be able to move the data folder from box.com into the same directory as the notebook that you are running.

You may need to utilize the rename notebook in "Data Processing/Utils" in order to get the data in the correct format.


# Files

* [Data Processing](https://github.com/lisaxu/CS598_Sleep_Stage_Classification/tree/main/Data%20Processing "Data Processing") - Contains examples and utilities that were utilized for data processing.
	* Utils/[rename_dir.ipynb](https://github.com/lisaxu/CS598_Sleep_Stage_Classification/blob/main/Data%20Processing/Util/rename_dir.ipynb "rename_dir.ipynb") - Used to rename data files numerically
	* [Databricks - Sleep Data Processing.html](https://github.com/lisaxu/CS598_Sleep_Stage_Classification/blob/main/Data%20Processing/Databricks%20-%20Sleep%20Data%20Processing.html "Databricks - Sleep Data Processing.html")
	* [Databricks - Sleep Data Processing.scala](https://github.com/lisaxu/CS598_Sleep_Stage_Classification/blob/main/Data%20Processing/Databricks%20-%20Sleep%20Data%20Processing.scala "Databricks - Sleep Data Processing.scala")
	* Examples
		* [Example read processed EEG data File.ipynb](https://github.com/lisaxu/CS598_Sleep_Stage_Classification/blob/main/Data%20Processing/Examples/Example%20read%20processed%20EEG%20data%20File.ipynb "Example read processed EEG data File.ipynb")
		* [sample.snappy.parquet](https://github.com/lisaxu/CS598_Sleep_Stage_Classification/blob/main/Data%20Processing/Examples/sample.snappy.parquet "sample.snappy.parquet")

 * [deliverables](https://github.com/lisaxu/CS598_Sleep_Stage_Classification/tree/main/deliverables "deliverables") - Contains the draft and proposal for the project

* [cnn-only.py](https://github.com/lisaxu/CS598_Sleep_Stage_Classification/blob/main/cnn-only.py "cnn-only.py") CNN only Model, training and Evaluation

 * [cnn-rnn.py](https://github.com/lisaxu/CS598_Sleep_Stage_Classification/blob/main/cnn-rnn.py "cnn-rnn.py") - CNN-RNN Model, training and Evaluation

* [SS_EDA_NonDL.ipynb](https://github.com/lisaxu/CS598_Sleep_Stage_Classification/blob/main/SS_EDA_NonDL.ipynb "SS_EDA_NonDL.ipynb") - Traditional Non-Deep Learning Models

# Dependencies

* Python 3
	* pandas
	* numpy
	* sklearn
	* icecream
	* pytorch
* Jupyter Notebook
