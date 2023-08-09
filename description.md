Your tasks will involve training and evaluating predictors for the Breast Cancer Wisconsin dataset. The dataset
contains features extracted from microscopic images of breast cancer cells along with a diagnosis: malignant (M) or
benign (B). Each feature describes the shape of cell nuclei in a sample taken from a breast tumor. You can find
the dataset along with more information about the features here:
[Breast Cancer Wisconsin (Diagnostic)](http://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
## Task 1
Build and train a model for predicting the dataset using Tensorflow and Keras. This model should contain at least one
custom layer. Please explain the structure of the model and how your custom layer works.
## Task 2
Write a callback for your model that reports batch losses on TensorBoard.
## Task 3
Train a different kind of model on the dataset. You are free to select the algorithm here.
Please explain how you selected the algorithm used in this task.
## Task 4
Evaluate and compare the performance
of the models created in the previous tasks. Interpret the results and select the better model from
the two.
## Task 5
If you observe the test set, would you consider it out-of-distribution compared to the train test? How would you
detect if the data generating distributions behind the train and the test set are different? Please provide a brief
explanation of your decision. It can be theoretical or concrete quick experimental code in a script / notebook.
## Task 6
Let's assume that we are ready to release an updated version of your proposed model into production. The team is
happy for the strong predictive performance of your model. Then, our Chief Scientific Officer asks questions about
the "why's" behind the model's prediction. How would you explain the chain of events inside your model of choice
that led to the predictions on the output. Let's assume, you have to explain the model's predictions on 10 arbitrarily
selected samples by one of our domain experts. Would you reconsider your choice of model class if interpretability is
a strong requirement? Which model class would you describe as the most interpretable of your proposed options?
# Task 7
Please provide an algorithmic method that quantifies the "excitingness / interestingness" of your model's
predictions. We assume that your model will be used for inference in a real-world scenario where the test set is
completely hold-out - true labels are unknown. Yet, we want to ensure that the model is capable of finding nontrivial
relationships between input features and labels. We say that a model is providing "exciting" predictions if it
capable of reaching a high level of predictive performance by finding more complex relationships in the dataset than
simply the distribution of true labels or linear mappings from features to true label. Provide a single scalar metric
that can be used to assess how much better your model is compared to very simple, linear models capable of finding
trivial biases in the data.
## Submission details
You must implement all tasks in
Python. Your submission should include all Python code (dataset loading, feature transformations,
training and evaluation) and a report containing results, explanations, and interpretations.
