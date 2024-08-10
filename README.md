# Customer-conversion-prediction---Insurance

Steps that I have done in this project:-
*First import the need packages,
*Load the dataset,
*Checking the null values in dataset,
*Remove Duplicate values,
*Checking the target whether it is equally distributed or not,
*Droping target variable from the dataset,
*Correlation matrix and Correlation heatmap for numerical and Categorical features,
*Outliers for numerical features,
*Splitng into training and test sets,
*Again spliting the training data into training and validation sets,
*Scaling the numerical features using pipeline,
*Encoding the Categorical features using pipeline,
*Preprocessor on the training data,validation data and test data,
*Over sampling(SMOTE) for training dataset.
*Trainig model and finding features importance of each model,
*We are finding the accuracy of training and test dataset 
*Finally Customer Segmentation based on Clusters.

**************************************************************************************************************

Note:-
If you want to see the plot of every procedure run as it is."python Insurance.py" in Visual Studio Code.
If you want see in the streamlit uncomment the last portion and uncomment the streamlit package and then run as "streamlit run Insurance.py".


Customer Clusting:-
Cluster 0:

Average Age: 40 years
Average Duration of Last Call: 300 seconds
Average Number of Calls: 2
Number of customers: 1500

Cluster 1:

Average Age: 35 years
Average Duration of Last Call: 250 seconds
Average Number of Calls: 1
Number of customers: 1200

Cluster 2:

Average Age: 50 years
Average Duration of Last Call: 400 seconds
Average Number of Calls: 3
Number of customers: 1000

Cluster 3:

Average Age: 45 years
Average Duration of Last Call: 350 seconds
Average Number of Calls: 2
Number of customers: 1300
