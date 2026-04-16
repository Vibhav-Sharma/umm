1. Linear Regression
Idea:

Draw a best-fit line.

👉 “Given X, predict a number Y”

Formula thinking:
y = m1*x1 + m2*x2 + ... + b
Use case:
Price prediction
Body fat
Sales
Viva line:

👉 “It minimizes squared error between actual and predicted values.”

🔵 2. Multiple Linear Regression

Same thing, just:
👉 more than 1 input feature

Example:

Area + Bedrooms + Age → Price
Difference:
Linear = 1 feature
Multiple = many features
🔴 3. Logistic Regression
Idea:

👉 Classification (NOT regression)

Output:

0 or 1
Yes/No

Uses sigmoid function

Example:
Spam detection
Pass/Fail
Viva line:

👉 “Uses sigmoid to map output between 0 and 1.”

🟢 4. KNN (K-Nearest Neighbors)
Idea:

👉 “Look at nearest neighbors and copy them”

Steps:

Pick K (like 3 or 5)
Find closest points
Majority vote
Example:
If 3 nearest are cats → it’s a cat
Viva line:

👉 “Lazy learning, no training phase.”

🟣 5. SVM (Support Vector Machine)
Idea:

👉 Draw best boundary between classes

It tries to:
👉 maximize distance between classes

Example:
Separate apples vs oranges
Viva line:

👉 “Finds optimal hyperplane with maximum margin.”

🌳 6. Decision Tree (ID3 / CART)
Idea:

👉 Tree of decisions

Example:

Is age > 30?
   yes → next condition
   no → next condition
ID3:
Uses Entropy
CART:
Uses Gini Index
Viva line:

👉 “Splits data recursively based on best feature.”

🌲 7. Random Forest
Idea:

👉 Many decision trees together

Each tree votes → final answer

Why better?
Reduces overfitting
Viva line:

👉 “Ensemble of decision trees using bagging.”

🟡 8. Naive Bayes
Idea:

👉 Probability-based

Uses:
👉 Bayes Theorem

Assumes:
👉 features are independent (naive)

Example:
Spam detection
Viva line:

👉 “Fast probabilistic classifier with independence assumption.”

🔵 9. K-Means (Clustering)
Idea:

👉 Group data into clusters

Steps:

Choose K clusters
Assign points
Update centers

Repeat.

Example:
Customer segmentation
Viva line:

👉 “Unsupervised algorithm using centroid-based clustering.”

🟣 10. PCA (Principal Component Analysis)
Idea:

👉 Reduce dimensions

Example:

100 features → 2 features
Why?
Faster
Less noise
Viva line:

👉 “Transforms data to lower dimensions while preserving variance.”

🔥 11. AdaBoost
Idea:

👉 Combine weak models → strong model

Focuses more on:
👉 wrong predictions

Viva line:

👉 “Boosting technique that adjusts weights of misclassified samples.”

⚡ 12. XGBoost
Idea:

👉 Advanced boosting (faster + better)

Why popular?
High accuracy
Used in competitions
Viva line:

👉 “Optimized gradient boosting algorithm with regularization.”

🧠 13. Self Organizing Map (SOM)
Idea:

👉 Map high-dim data → 2D grid

Used for:

Visualization
Clustering
Viva line:

👉 “Unsupervised neural network for dimensionality reduction.”

🎮 14. Q-Learning
Idea:

👉 Learning by rewards

Agent learns:

what action gives best reward
Example:
Game AI
Robotics
Viva line:

👉 “Reinforcement learning using Q-table for state-action values.”
