## Project Description

We will work with a dataset containing rental property listings from a German real estate platform. The dataset includes details like living area size, rent, location (street, house number, ZIP code, state), type of energy, and more. Importantly, it features two fields with longer free text: a "description" of the offer and "facilities" detailing available amenities and recent renovations. This dataset is accessible at Kaggle.

The project is divided into two main parts, focusing on machine learning model development for rent prediction:

-Predicting Rent with Structural Data: Develop a machine learning model to predict the total rent using only the structural data. For this, we exclude the “description” and “facilities” text fields for this model.

-Predicting Rent with Structural and Text Data: We create a second machine learning model that predicts the total rent using both structural and text data (“description” and “facilities”). For this part, we use Generative AI techniques for processing text data.

----------------------------------------------

## Read Me

For this project, I first started applying Generative AI to the text variables ('description' and 'facilities') to derive other variables from a sentiment analysis.
The chosen variables were: luxury, accessibility, security, safety (proxys of each other), tranquility, overall_sentiment.
For this, I developed a code to interact with the OpenAI API, creating my own Key and adding some funds in order to use its full power. The queries took approximately 1.5 days to process more than 90% of the data, where I stopped.
I later added these variables to the main dataframe.
Given the semantic nature of the interaction with OpenAI through prompts, out of the hundreds of thousands of results yielded, some were mistaken or with inadequate format. I had to manually filter and clean the dataset to make it available for work and to prevent further format issues down the line.

With respect to the prediction models build up, my approach was, besides achieving good accuracy, to spare computation time and power and to make a more meaningful model easier to generalize later on.
For this purpose I carried out a thorough examination of every variable, trying to understand its nature, distribution and characteristics. I removed several variables that were not optimal, or remarkably beneficial for the model, increasing complexity without much added value. Many variables were proxys of each other, like those which ended as 'Range', since most of their information was stored in the original, non-ranged variables.

After that, I studied outliers individually per variable, both setting the filters manually and applying the interquartile range criteria, depending on the distribution shape in the histograms, and which impact the decision would have on the distribution. 
I later applied several encodings to the categorical variables. For the target encoding it was mandatory to have a clean and comprehensible target variable, free of its extreme outliers that misshaped its distribution. In the same manner as before, I evaluated which of the variables had the right conditions for each type (booleans, cardinality,...). 
Then, I filled the NaNs and proceeded to introduce them in the model.

I chose an XGBoost model because, in my experience, usually offers great flexibility and can outperform other models. It combines several features of some of the most prominent model families in Machine Learning.
It also offers great tolerance to NaNs and outliers, which allows for a faster implementation, with less data alterations and engineering than other models.

I carried out several searches with RandomSearchCV and later tunned several hyperparameters with GridSearchCV. I believe this is definitely one of the greatest aspects of improvement, by covering a much greater amount of parameter combinations and adding more iterations. However, the results were very satisfactory on the test set, after having added 10 cross-validation folds.

Model 1 (strictly the one with only structural variables) did worse than Model 2 by evaluating the R² coefficient (0.90 to 0.92). However, in MAE (mean absolute error), the results were very even and even slightly favorable to Model 1.
A possible reason for this is due to the weight and sparsity of the high totalRent outliers, where predictions tend to fail in both cases.

##################################################
##### NOTEBOOKS & SCRIPT ######

All the developments are gathered in the Jupyter notebooks.

* SentimentAnalysis_script.ipynb is the code and proccess I followed to develop the prompt and query to the Open AI API. Not something that needs to be ran. The dataset has already been generated, its called 'sentiment_analysis_finalversion.txt'.

* Prediction_script.ipynb can be ran from top to bottom without a problem. It's the one where the models are trained, as well as the plot.

* script_project_AlbertoAlvarezSaavedra.py is a summarized python script to show on terminal the fit of both models to the dataset. 
Each models uses an adapted version of the dataset (df_model1, df_model2). These datasets should be decompressed (too heavy for manual upload to GitHub) and set in the same folder as the script.
It is also necessary for model files to be in the folder indicated in the script (XGB_models). 

You can just run it from terminal with:

python3 script_project_AlbertoAlvarezSaavedra.py

I recommend a python version >= 3.9
