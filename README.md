# Beat-the-market
Deep learning project carried out to predict stock price trends as part of my studies in Reykjavik.

The model.py file contains the neural network itself.
The .pt file contains the weights recorded after training.
The .pkl files contain the data respectively from the normalisation and the PCA carried out when the network was trained on the data.
And the simulation, prediction and main files are the files used to carry out the prediction.

Here is the link to neural network training:
https://colab.research.google.com/drive/1KqZ1pT7ZYKMJNm22lbHPQVDKCH5AyRBk?usp=sharing

Everything needs to be extracted together in the same place,
the predictor.py calls model.py and pmodel.pt
the pca.pkl and scaler.pkl are needed to scale and transform data

After extracting at the right place (where main, data, simulator are located), 
the folder should look like:

	/data
	main.py
	model.py
	pca.pkl
	pmodel.pt
	predictor.py
	scaler.pkl
	simulator.py
