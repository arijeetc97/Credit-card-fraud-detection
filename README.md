# Credit-card-fraud-detection

Dataset taken from https://www.kaggle.com/mlg-ulb/creditcardfraud

**Description**

There was a huge imbalance in the dataset, frauds only account for 0.172% of fraud transactions.
In this case, it is much worse to have false negatives than false positives in our predictions because false negatives mean that someone gets away with credit card fraud. False positives, on the other hand, merely cause a complication and possible hassle when a cardholder must verify that they did, in fact, complete said transaction (and not a thief).

One challenging aspect of the dataset was that there were 30 features, but inorder to protect confidentiality, all but 28 of them had been PCA transformed and had unknown labels. The known, non-transformed features were 'Time', which measures the seconds between the transaction and the first transaction in the 2-day time period, and 'Amount', which is the cost of the transaction, presumably in Euros.
In order to offset the imbalance in the dataset, I oversampled the fraud (class = 1) portion of the data, adding Gaussian noise to each row.

In terms of the neural network, performing principal component analysis on the oversampled data before splitting it into training and test sets resulted in a jump from 50% accuracy to 94.56% accuracy. Before PCA, nothing we tried was able to push the accuracy past 50%. After this step, however, adjustments to the number of layers, activation functions, and neurons in each layer did not do much to change the accuracy, which hovered at just below 95%. Furthermore, choosing only 2 neurons for the first dense layer (called a "bottleneck effect") forced the model to really reduce to only the most necessary features and decrease the likelihood of overfitting.

While the neural network had a high accuracy, its biggest pitfall was that within its 5.44% inaccuracy rate, 84.64% were false negatives. The fact that this neural network missed 4.60% of frauds is enough to make this model infeasible compared to the other methods, where the false negative rate was much lower. Interestingly, a switch from a sigmoid to tanh activation function reduced the false negative rate by about 1%.
