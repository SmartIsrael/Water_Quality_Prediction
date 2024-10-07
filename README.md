# Water Quality Model - Group 5

## Project Overview
This project aims to build a machine learning model to analyze water quality data and predict water quality. The project includes various stages such as data handling, model training, regularization, and evaluation. Here is the link to the Colab Notebook for viewing: https://colab.research.google.com/drive/1bblWAGeYnDQahTGlLRSA7eiS6jDhAgeX?usp=sharing

### GROUP STRUCTURE AND TASK ALLOCATION

1. Data Handler- Smart Israel
   **Role:** Responsible for loading the dataset and data preprocessing(cleaning, splitting the data, and scaling) it for training.
   
2. Vanilla Model Implementor - Teniola Ajani
   **Role:** Vanilla Model Implementation

3.  L1 Regularization Implementor - Emmanuel Begati
   **Role:** Focuses on applying L1 Regularization to the model and testing it's effect on the performance.
    
4. L2 Regularization Implementor - Kathrine Ganda
   **Role:** Responsible for applying L2 Regularization to the model and testing its effect on the performance. Comparing its results to the vanilla model and L1 regularization.

5. Error Analysis - Anissa
   **Role:** Performs analysis on errors made by the models.

# Data Handling and Preprocessing in Water Quality Analysis Project

### Overview

This documentation covers the data handling and preprocessing steps in the Water Quality Analysis project. These steps are crucial for preparing the data for machine learning models, ensuring that the data is clean, properly formatted, and optimized for analysis.

### 1. Data Loading

The first step in our data handling process is loading the data from a CSV file.

```python
import pandas as pd

data_csv = './water_potability.csv'
df = pd.read_csv(data_csv)
```

**Explanation:**
- We use pandas, a powerful data manipulation library in Python.
- The `read_csv()` function loads the data from the CSV file into a pandas DataFrame.
- The resulting `df` is our main DataFrame that we'll work with throughout the preprocessing steps.

### 2. Data Exploration

Before preprocessing, it's important to understand our data:

```python
df.head()
df.describe()
df.shape
df.isnull().sum()
```

**Explanation:**
- `df.head()`: Displays the first few rows of the DataFrame, giving us a quick look at the data structure.
- `df.describe()`: Provides statistical summary of the numerical columns (count, mean, std, min, 25%, 50%, 75%, max).
- `df.shape`: Returns a tuple representing the dimensionality of the DataFrame (rows, columns).
- `df.isnull().sum()`: Counts the number of null values in each column.

### 3. Handling Missing Values

Our approach to handling missing values is to replace them with the mean of their respective columns:

```python
df = df.fillna(df.mean())
```

**Explanation:**
- `fillna()` is used to fill null values.
- We use the mean of each column to fill the nulls in that column.
- This method helps preserve the overall distribution of the data while filling in the gaps.

**Note:** The choice to use mean imputation was made to maintain the dataset size. However, it's important to consider that this method can reduce the variance in the data and potentially weaken relationships between variables. In a more comprehensive analysis, you might consider more advanced imputation techniques or analyze the pattern of missingness.

### 4. Feature Selection

We separate our features (X) from our target variable (y):

```python
X = df.drop('Potability', axis=1)  # Features
y = df['Potability']  # Target variable
```

**Explanation:**
- `df.drop('Potability', axis=1)` creates a new DataFrame `X` with all columns except 'Potability'.
- `df['Potability']` selects only the 'Potability' column as our target variable `y`.

### 5. Data Splitting

We split our data into training and testing sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Explanation:**
- We use scikit-learn's `train_test_split` function.
- `test_size=0.2` means 20% of the data will be used for testing, and 80% for training.
- `random_state=42` ensures reproducibility of the split.

### 6. Feature Scaling

The final preprocessing step is scaling our features:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Explanation:**
- We use `StandardScaler` from scikit-learn, which standardizes features by removing the mean and scaling to unit variance.
- `scaler.fit(X_train)`: The scaler learns the mean and variance of the training data.
- `scaler.transform(X_train)`: The training data is transformed using the learned parameters.
- `scaler.transform(X_test)`: The test data is transformed using the same parameters learned from the training data.

**Important:** We fit the scaler only on the training data to prevent data leakage. The same transformation is then applied to both training and test data.

### Conclusion

These preprocessing steps prepare our water quality data for machine learning models. The data is now cleaned, split into training and testing sets, and scaled appropriately. This preprocessing pipeline ensures that our data is in the optimal format for training our neural network models, including those with L1 and L2 regularization.

---

# Vanilla Model Implementation

   - Built a baseline model with three layers: input layer(64 neurons), two hidden layers (32 neurons and 64 neurons), and an output layer (1 neuron).
   - Used ReLU activation for the input and hidden layers and sigmoid activation for the output layer for binary classification
   - Trained the model for 100 epochs with 20% validation split and a batch size of 32.
   - Achieved a test accuracy of **65.5%**, but overfitting was observed, indicating the need for regularization.
    ![loss](https://github.com/SmartIsrael/Water_Quality_Prediction/blob/main/Vanilla_model_loss.png)
    
---

# L1 Regularization Implementation

L1 regularization, also known as Lasso regularization, was applied to the neural network model to encourage sparsity and potentially improve generalization.

### Model Architecture

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l1

l1_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],),
          kernel_regularizer=l1(0.001)),
    Dense(32, activation='relu', kernel_regularizer=l1(0.001)),
    Dense(1, activation='sigmoid')
])
```

- L1 regularization is applied to both hidden layers with a regularization strength of 0.001
- The output layer uses sigmoid activation for binary classification

### Model Compilation and Training

```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

l1_model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

l1_early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

l1_model_fitted = l1_model.fit(X_train_scaled, y_train,
                               epochs=50,
                               batch_size=32,
                               validation_split=0.2,
                               callbacks=[l1_early_stopping],
                               verbose=1)
```

- Adam optimizer was used with a learning rate of 0.001
- Early stopping was implemented to prevent overfitting
- The model was trained for a maximum of 50 epochs with early stopping

## Results

The L1 regularized model's performance was evaluated on the test set:

```python
loss, test_accuracy, test_precision, test_recall, test_auc = l1_model.evaluate(X_test_scaled, y_test, verbose=1)

print(f'L1 Model Test Accuracy: {test_accuracy}')
print(f'Test Loss: {loss}')
print(f'L1 Model Test Precision: {test_precision}')
print(f'L1 Test Recall: {test_recall}')
print(f'L1 Test AUC: {test_auc}')
```

21/21 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - AUC: 0.6861 - Precision: 0.5969 - Recall: 0.3468 - accuracy: 0.6827 - loss: 0.6301 
L1 Model Test Accuracy: 0.6829268336296082
Test Loss: 0.6328526735305786
L1 Model Test Precision: 0.6285714507102966
L1 Test Recall: 0.3606557250022888
L1 Test AUC: 0.6949506998062134


## Note on Optimizer Choice

While Adam optimizer was used in the final implementation, RMSprop was also tested. However, Adam provided better results for this particular dataset and model architecture.

## Conclusion

The L1 regularization technique was applied to improve the model's generalization capability. While it helped reduce overfitting to some extent, the overall performance improvement was modest. Further tuning of hyperparameters or exploration of other techniques might be necessary for significant improvements.


# L2 Regularisation Implementation - Katherine Ganda

This technique was applied to the vanilla model neural network to prevent overfitting. The Adam optimiser which provided slightly better results with an accuracy of (**68.65%**) against RMSprop**(68.45%)**and early stopping were used to improve the model’s performance further.

To find the most effective regularization strength, several L2 values were tested, aiming for stable model performance and good generalization.

Values tested: 0.1,0.005, 0.01

**L2 Parameter(0.005)**

![loss1](https://github.com/user-attachments/assets/dc70a046-cbf5-45a3-a8ff-27d1c3c1e29c)

At a regularization value of 0.005, the training loss continues to decrease, but the validation loss begins to fluctuate after epoch 5. The widening gap between the two losses indicates overfitting, suggesting that this regularization strength is not stable for generalization.

**L2 Parameter(0.1)**

![loss 3](https://github.com/user-attachments/assets/3aa1b08b-a69e-4d94-8e47-cd9147954fa1)

With a higher L2 regularization value of 0.1, both training and validation losses drop rapidly and converge by epoch 5. However, this may indicate underfitting due to the stronger regularization, as the model stops learning early, showing minimal improvement beyond epoch 5.

**L2 Parameter(0.01)**

![loss2](https://github.com/user-attachments/assets/7a7647eb-c0ee-4285-ab3a-bb10d0249598)

At L2 = 0.01, both training and validation losses decrease smoothly and converge closely. The small gap between the two losses and the absence of significant validation loss fluctuations suggest that this regularization value achieves good generalization and model stability.

After comparing the different values, L2 regularization with a parameter of 0.01 provided the most stable performance, balancing generalization and convergence, with minimal gap between training and validation losses and was therefore chosen for the task.

# Comparison L1 Regularisation and L2 Regularisation

**L1:**
![image](https://github.com/user-attachments/assets/508c0d5d-eba1-4319-b875-ca2fef729685)

**L2**
![image](https://github.com/user-attachments/assets/5aa607e5-f622-4f5b-8316-9a879077c34d)

From the model’s evaluation, upon comparing the performance, L2 regularisation provided slightly better performance compared to l1 with both a higher accuracy and lower loss. These results suggest that L2 regularisation helped the model generalize better and improved its ability by minimizing the errors.

## Explanation of Results

**1. Feature Retention:**

In a relatively small dataset with 9 features, all features likely contribute meaningfully to the predictions. L2 regularization shrinks the weights without reducing any to zero, allowing the model to retain all feature contributions. In contrast, L1 regularization drives some weights to zero, effectively removing certain features from the model. This can be beneficial for large and complex datasets but may lead to the omission of valuable features in smaller datasets, resulting in slightly lower performance.

**2. Simple Network Structure:**

The network, with only two hidden layers, is relatively simple. In such cases, L1 regularization is not as effective for feature selection because the model does not have sufficient complexity to benefit from eliminating features. L2 regularization’s smoother reduction of weights allows all features to be utilized, improving performance by maintaining their contributions. In more complex models, L1 might be more advantageous, but for this simple architecture, L2 proved to be more suitable.

## Conclusion:


## **Error Analysis - Anissa OUEDRAOGO**

As part of the evaluation of our water quality prediction models, error analysis was conducted to compare the performance of three models: **Vanilla (no regularization)**, **L1 Regularization**, and **L2 Regularization**. The primary metrics used for error analysis were:

- **Mean Squared Error (MSE):** Measures the average squared difference between actual and predicted values.
- **Mean Absolute Error (MAE):** Captures the average absolute difference between actual and predicted values.
- **Root Mean Squared Error (RMSE):** Provides the square root of the MSE, maintaining the units of the target variable.

  ![image](https://github.com/user-attachments/assets/81e9bdb1-b2a7-4d1e-aa0d-31a50ac52558)

### 1. **Vanilla Model:**
- **MSE:** 0.2517
- **MAE:** 0.3995
- **RMSE:** 0.5017
- **Observations:** The vanilla model exhibited the highest error across all metrics. This indicated **overfitting** to the training data, highlighting the need for regularization to improve generalization on unseen data.

### 2. **L1 Regularization Model:**
- **MSE:** 0.2037
- **MAE:** 0.4077
- **RMSE:** 0.4514
- **Observations:** The L1 regularized model showed improved performance over the vanilla model in terms of MSE and RMSE, though it slightly increased the MAE. **L1 regularization** tends to drive some feature weights to zero, reducing model complexity, which may result in improved generalization but also slight feature elimination.

### 3. **L2 Regularization Model:**
- **MSE:** 0.2026
- **MAE:** 0.4101
- **RMSE:** 0.4501
- **Observations:** The L2 regularized model performed best with the lowest MSE and RMSE. **L2 regularization** shrinks the weights of less important features without driving them to zero, resulting in a more balanced and generalizable model. This model showed the greatest ability to generalize well on the test set.

### **Conclusion:**
The error analysis demonstrated that both **L1** and **L2 regularization** techniques improved the model’s generalization ability over the vanilla model. However, **L2 regularization outperformed L1 regularization** slightly in this case, as it maintained better performance across all metrics. Regularization effectively controlled overfitting, allowing the models to perform better on unseen data.

