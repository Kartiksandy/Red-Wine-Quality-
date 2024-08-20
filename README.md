## Red Wine Quality Classifier using a Neural Network

This project involves building and evaluating a neural network model to classify the quality of red wine based on various chemical properties. The model is developed using TensorFlow's Estimator API and Keras, and it is trained and tested on a dataset containing chemical properties of red wine.

### Project Overview:

1. **Objective:**
   - The primary objective is to predict the quality of red wine using a neural network model based on various chemical properties.

2. **Data Preprocessing:**
   - **Loading the Dataset:** The wine quality dataset is loaded from a CSV file.
   - **Normalization:** The feature columns are normalized using `StandardScaler` to ensure the inputs to the model have similar scales.
   - **Splitting the Data:** The dataset is split into training and testing sets using an 80-20 split.

   ![Histogram of Features](path_to_histogram_plot.png)
   ![Correlation Matrix](path_to_correlation_matrix_plot.png)

3. **Model Building:**
   - **Feature Columns:** Defined numeric feature columns for each input feature using `tf.feature_column.numeric_column`.
   - **DNN Classifier:** A Deep Neural Network (DNN) classifier is built using the TensorFlow Estimator API with two hidden layers (30 and 10 units respectively).
   - **Input Functions:** Input functions are defined for both training and evaluation using `tf.data.Dataset`.

4. **Neural Network using Keras:**
   - An alternative implementation of the neural network model is built using the `tf.keras` API, providing flexibility in model customization and training.

5. **Training and Evaluation:**
   - **Training with TensorBoard:** The model is trained for 5000 steps with TensorBoard logging enabled to visualize the training process.
   - **Evaluation:** The model’s performance is evaluated on the test set, and the test accuracy is reported.

   ![Training and Validation Loss](path_to_training_validation_loss_plot.png)
   ![Training and Validation Accuracy](path_to_training_validation_accuracy_plot.png)

6. **Key Findings and Analysis:**
   - **Data Normalization:** Ensures that each input feature contributes equally during training, preventing larger-scale features from dominating the learning process.
   - **Model Architecture:** The chosen architecture with two hidden layers balances model complexity and computational efficiency.
   - **Training Process:** TensorBoard visualizations provided insights into how loss and accuracy evolved over time, aiding in diagnosing potential issues such as overfitting.
   - **Evaluation:** Test accuracy gives a measure of how well the model generalizes to unseen data, crucial for real-world performance.

### How to Use:

1. **Clone the Repository:**
   - Clone this repository to your local machine using `git clone`.
   
2. **Install Dependencies:**
   - Install the required Python packages using `pip install -r requirements.txt`.

3. **Run the Notebook:**
   - Open the notebook in Jupyter and execute the cells in sequence to preprocess the data, build the model, and evaluate its performance.

### Visual Examples:

- **Feature Histograms:** 
  ![Histogram of Features](path_to_histogram_plot.png)
  
- **Correlation Matrix:** 
  ![Correlation Matrix](path_to_correlation_matrix_plot.png)
  
- **Training and Validation Loss:** 
  ![Training and Validation Loss](path_to_training_validation_loss_plot.png)
  
- **Training and Validation Accuracy:** 
  ![Training and Validation Accuracy](path_to_training_validation_accuracy_plot.png)

### Conclusion:

This project demonstrates the effective use of neural networks for predicting red wine quality based on chemical properties. The model’s accuracy and insights gained from TensorBoard visualizations highlight the importance of proper data preprocessing, model architecture selection, and evaluation techniques in machine learning projects.
