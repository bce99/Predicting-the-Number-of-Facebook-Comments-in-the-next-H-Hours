# Predicting-the-Number-of-Facebook-Comments-in-the-next-H-Hours
Coursework project under NUS CS3244 Machine Learning
### Source Code Descriptions:  

**1 - Data Preprocessing.ipynb**:   
		- Preliminary preparations of the dataset, including differentiating the statistical columns (4-29) to compare results and normalization of the dataset   
		- Correlation matrix and its visualization to explore between-feature relationships - Fitting OLS model, subsequently generating PCA components using the original data set   
	
**2 - corrMatrix1.xlsx**:   
		- Correlation matrix of the columns in the data, with highly correlated pairs highlighted in red   
	
**3 - Linear Regression [All Models].ipynb**:   
	- Converting data to dummy variables and normalizing data, according to the process in *1 - Data Preprocessing.ipynb*     
	- The implementation of Linear Regression with different variations of the data (by dropping columns)     
	- The implementation of Linear Regression with principal component analysis and stochastic gradient descent     
	
**4 - Linear Regression [Selection of Features for p-value, SelectKBest].ipynb**:  
	- Deriving the significant features in the data based on p-value     
	- Using SelectKBest to find K best features   
	- These derived features were used to build models in *3 - Linear Regression [All Models].ipynb*     
	
**5 - Decision Tree.ipynb**:   
	- Testing the relationship between MSE/R-squared against the maximum depth of the regressor tree   
	- Using ensemble methods to improve the performance of prediction   
	- Implementing 10-fold HalvingGridSearchCV on the improved models to generate the optimal max_depth and max_features combination   
	- Selecting significant features by visualizing decision tree / comparing root-mean-squared values for explainability     
	
**6 - MLP [Sklearn].ipynb**:   
	- Implementing MLP with scikit-learn   
	- Using GridSearchCV with provided parameters to look for best parameters - Fitting the MLP model with best parameters and evaluating the performances with r2 score and mean squared error    
	
**7 - best_MLP.pt**:   
	- The trained weights which yield the highest performance   
	- Can be loaded in *6 - MLP [PyTorch].ipynb* to reproduce test results and MLP explainability results mentioned in the video   
	
**8 - MLP [PyTorch].ipynb**:  
	- The implementation of MLP using PyTorch, including both training and testing  
	- The implementation of MLP explainability techniques  

### References mentioned in our presentation video:   
#### Existing work on the dataset   
● Wu, S. (n.d.). WUSIXUAN2/Facebook-Comment-volume-prediction. GitHub. Retrieved November 18, 2022, from 
https://github.com/wusixuan2/facebook-comment-volume-prediction   
● Singh, K., Kumar, D., & Kaur, R. (03 2015). Comment Volume Prediction using Neural Networks and Decision Trees. doi:10.1109/UKSim.2015.20   
#### Other materials   
● Gilde, K. (2021, January 16). A faster hyper parameter tuning using nature-inspired algorithms in ... Faster Hyperparameter Tuning with Scikit-Learn’s HalvingGridSearchCV. Retrieved November 18, 2022, from
https://towardsdatascience.com/a-faster-hyper-parameter-tuning-using-nature-inspired-algorithms-in-python-33a32eb34f54   
● Rob Hyndman (n.d.). How to choose the number of hidden layers and nodes in a feedforward neural network? Cross Validated. Retrieved November 18, 2022, from https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw   
● Molnar, C. (2022, November 12). Interpretable machine learning. 9.6 SHAP (SHapley Additive exPlanations). Retrieved November 18, 2022, from 
https://christophm.github.io/interpretable-ml-book/shap.html   
● Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.   
● Lee, W.-M. (2022, January 31). Using principal component analysis (PCA) for Machine Learning. Using Principal Component Analysis (PCA) for Machine Learning. Retrieved November 18, 2022, from
https://towardsdatascience.com/using-principal-component-analysis-pca-for-machine-learning-b6e803f5bf1e   
● Lars Buitinck, Gilles Louppe, Mathieu Blondel, Fabian Pedregosa, Andreas Mueller, Olivier Grisel, Vlad Niculae, Peter Prettenhofer, Alexandre Gramfort, Jaques Grobler, Robert Layton, Jake VanderPlas, Arnaud Joly, Brian Holt, & Gaël Varoquaux (2013). API design for machine learning software: experiences from the scikit-learn project. In ECML PKDD Workshop: Languages for Data Mining and Machine Learning (pp. 108–122).   
● Solanki, S. (2022, August 6). How to use lime to interpret predictions of ML models [python]? by Sunny Solanki. Developed for Developers by Developer for the betterment of Development. Retrieved November 18, 2022, from 
https://coderzcolumn.com/tutorials/machine-learning/how-to-use-lime-to-understand-sklearn-models-predictions   
● Cohen, I. (2021, May 23). Explainable AI (XAI) with shap - regression problem. Explainable
AI (XAI) with SHAP - regression problem. Retrieved November 18, 2022, from https://towardsdatascience.com/explainable-ai-xai-with-shap-regression-problem-b2d63fdca670   
● towardsdatascience. (2020, September 20). When and why tree-based models (often) outperform neural networks. When and Why Tree-Based Models (Often) Outperform Neural Networks. Retrieved November 18, 2022, from 
https://towardsdatascience.com/when-and-why-tree-based-models-often-outperform-neural-networks-ceba9ecd0fd8   
● Regression Example with SGDRegressor in Python. (2020, September 15). https://www.datatechnotes.com/2020/09/regression-example-with-sgdregressor-in-python.html   
● Biswal, A. (2022, November 15). Sklearn Linear Regression. Simplilearn.com. https://www.simplilearn.com/tutorials/scikit-learn-tutorial/sklearn-linear-regression-with-examples   
● UPDATE: Converting Python Data to R With RPY2 - Remy Canario. (2021, December 13). https://medium.com/@remycanario17/update-converting-python-dataframes-to-r-with-rpy2-59edaef63e0e  
