# Real Estate Price Prediction

📌 Description

The Real Estate Price Prediction system leverages advanced machine learning algorithms to predict property prices based on various features. By combining multiple regression models, it aims to provide accurate price projections for homes, crucial for buyers, sellers, and investors in the dynamic real estate market. Real-time price predictions are calculated using up to 20 different parameters derived from a comprehensive dataset of Boston housing prices.

The project applies ensemble learning techniques, using models like AdaBoost Regressor, SVR, and Random Forest Regressor, and integrates a chatbot for interactive price predictions based on user input.

🚀 Features

	•	Multiple Models: Combines different machine learning models to improve accuracy, including AdaBoost, SVR, Random Forest, and more.
 
	•	Ensemble Learning: Uses stacking techniques to combine predictions from multiple models for more accurate results.
 
	•	Interactive Chatbot: Allows users to predict house prices through voice or text input.
 
	•	Real-Time Predictions: Generates real-time property price predictions based on specific features entered by the user.
 
	•	Data Analysis & Visualization: Includes detailed exploration of correlations between features using heatmaps.
 
	•	Hyperparameter Tuning: Optimizes model performance through fine-tuning.

🛠️ Tech Stack

	•	Programming Language: Python
 
	•	Libraries: Pandas, Scikit-learn, Matplotlib, Seaborn, OpenAI API, SpeechRecognition
 
	•	Machine Learning: Linear Regression, Decision Tree, Random Forest, AdaBoost, Support Vector Machine (SVM), KNN, Ensemble Learning
 
	•	Data Visualization: Heatmaps, Scatter Plots
 
	•	Chatbot Integration: OpenAI API, Text-to-Speech, Speech-to-Text

🔥 Usage

The user can interact with the chatbot by either speaking or typing queries related to house price predictions.

For example:

	•	The chatbot will ask for inputs like the number of rooms, tax rate, crime rate, etc.
 
	•	Based on the input, it predicts the house price in real-time.

 📝 Data Preprocessing

The dataset consists of 506 entries with 14 parameters affecting house prices. The preprocessing steps include:

	•	Replacing missing values with the median of existing values.
 
	•	Combining certain attributes (e.g., TAX and RM) to enhance model efficiency.

📊 Data Analysis

Each parameter’s correlation with others is analyzed using a heatmap to check for relationships that might affect house prices. A higher absolute correlation indicates a stronger relationship between parameters.

⚙️ Model Training & Evaluation

	•	Training: The data is split into training and testing sets (80/20). The training process uses AdaBoost Regressor, SVR, and RandomForest Regressor.
 
	•	Evaluation: The models are evaluated based on their Root Mean Squared Error (RMSE) and accuracy.
 
	•	Ensemble Learning: A stacking technique is used to combine predictions from multiple models, which improves the RMSE significantly compared to individual models.

💬 Chatbot Integration

	•	The chatbot listens for user queries using speech recognition or text input.
 
	•	It processes the features, makes predictions, and provides feedback on the predicted house price using a trained machine learning model.

💡 Acknowledgements

	•	Special thanks to Carnegie Mellon University for the Boston Housing Dataset.
 
	•	Thanks to the Scikit-learn and OpenAI API for providing the necessary tools to implement machine learning and chatbot functionality.
