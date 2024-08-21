## Red Wine Quality Classifier using a Neural Network

This project involves building and evaluating a neural network model to classify the quality of red wine based on various chemical properties. The model is developed using TensorFlow's Estimator API and Keras, and it is trained and tested on a dataset containing chemical properties of red wine.

### Project Overview:

1. **Objective:**
   - The primary objective is to predict the quality of red wine using a neural network model based on various chemical properties.

2. **Data Preprocessing:**
   - **Loading the Dataset:** The wine quality dataset is loaded from a CSV file.
   - **Normalization:** The feature columns are normalized using `StandardScaler` to ensure the inputs to the model have similar scales.
   - **Splitting the Data:** The dataset is split into training and testing sets using an 80-20 split.

3. **Model Building:**
   - **Feature Columns:** Defined numeric feature columns for each input feature using `tf.feature_column.numeric_column`.
   - **DNN Classifier:** A Deep Neural Network (DNN) classifier is built using the TensorFlow Estimator API with two hidden layers (30 and 10 units respectively).
   - **Input Functions:** Input functions are defined for both training and evaluation using `tf.data.Dataset`.

4. **Neural Network using Keras:**
   - An alternative implementation of the neural network model is built using the `tf.keras` API, providing flexibility in model customization and training.

5. **Training and Evaluation:**
   - **Training with TensorBoard:** The model is trained for 5000 steps with TensorBoard logging enabled to visualize the training process.
   - **Evaluation:** The model’s performance is evaluated on the test set, and the test accuracy is reported.

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
  
<img width="753" alt="image" src="https://github.com/user-attachments/assets/56b4f384-14e1-486f-bc60-37d305d54cc2">

  
- **Correlation Matrix:** 

<img width="751" alt="image" src="https://github.com/user-attachments/assets/496fc0a7-848f-4485-a632-9998b1701b84">

  
- **Training and Validation Loss:** 

<img width="726" alt="image" src="https://github.com/user-attachments/assets/d82bb8f5-67c9-4f06-b6f5-e0f12c35589c">

  
- **Training and Validation Accuracy:** 

<img width="722" alt="image" src="https://github.com/user-attachments/assets/45de2605-ec9a-43fd-9a87-d38497053fc5">


### Conclusion:

This project demonstrates the effective use of neural networks for predicting red wine quality based on chemical properties. The model’s accuracy and insights gained from TensorBoard visualizations highlight the importance of proper data preprocessing, model architecture selection, and evaluation techniques in machine learning projects.
