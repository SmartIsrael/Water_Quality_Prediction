Notebook: https://colab.research.google.com/drive/1bblWAGeYnDQahTGlLRSA7eiS6jDhAgeX?usp=sharing

# GROUP STRUCTURE AND TASK ALLOCATION

1. Data Handler- Smart Israel

   **Role:** Responsible for loading the dataset and data preprocessing(cleaning, splitting the data, and scaling) it for training.

2. Vanilla Model Implementor - Teniola Ajani

   **Role:** Implement a baseline Neural network model with a simple architecture(no regularization or optimization techniques).

3. L1 Regularization Implementor - Emmanuel Begati

   **Role:** Focuses on applying L1 Regularization to the model and testing it's effect on the performance.

4. L2 Regularization Implementor - Kathrine Ganda

   **Role:** Responsible for applying L2 Regularization to the model and testing its effect on the performance. Comparing its results to the vanilla model and L1 regularization.

5. Error Analysis - Anissa
   **Role:** Performs analysis on errors made by the models.

# L2 Regularisation Implementation

This technique was applied to the vanilla model neural network to prevent overfitting. The Adam optimiser and early stopping were used to further improve the model’s performance.

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
![L1_evaluation](https://github.com/user-attachments/assets/bf48cb14-0bb4-4f12-8a80-4a40551d9394)

**L2**
![L2_evaluation](https://github.com/user-attachments/assets/143940ae-73a5-45a6-9df3-2c97f2b53fbe)

From the model’s evaluation, upon comparing the performance, L2 regularisation provided slightly better performance compared to l1 with both a higher accuracy and lower loss. These results suggest that L2 regularisation helped the model generalize better and improved its ability by minimizing the errors.

## Explanation of Results

**1. Feature Retention:**

In a relatively small dataset with 9 features, all features likely contribute meaningfully to the predictions. L2 regularization shrinks the weights without reducing any to zero, allowing the model to retain all feature contributions. In contrast, L1 regularization drives some weights to zero, effectively removing certain features from the model. This can be beneficial for large and complex datasets but may lead to the omission of valuable features in smaller datasets, resulting in slightly lower performance.

**2. Simple Network Structure:**

The network, with only two hidden layers, is relatively simple. In such cases, L1 regularization is not as effective for feature selection because the model does not have sufficient complexity to benefit from eliminating features. L2 regularization’s smoother reduction of weights allows all features to be utilized, improving performance by maintaining their contributions. In more complex models, L1 might be more advantageous, but for this simple architecture, L2 proved to be more suitable.

## Conclusion:

L2 regularization performed better by maintaining the contributions of all features. On the other hand, L1 regularization may have been too aggressive in eliminating potentially important features, leading to slightly worse performance.
