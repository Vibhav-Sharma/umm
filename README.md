🔵 1. KNN (K-Nearest Neighbors)
🧠 Idea

👉 “Similar things are nearby”

Use
Classification (mostly)
Simple problems
Example

Predict if a fruit is apple/orange based on nearest points

⚠️ When to use
Small dataset
Simple decision boundary
❌ Avoid when
Large dataset (slow)
🟣 2. SVM (Support Vector Machine)
🧠 Idea

👉 Best boundary between classes

Use
High accuracy classification
Works well in complex spaces
Example

Cancer detection

⚠️ When to use
Clear margin between classes
🌳 3. Random Forest
🧠 Idea

👉 Many decision trees → voting

Use
Almost any classification problem
Very reliable
Example

Fraud detection

🔥 Why good?
Reduces overfitting
⚡ 4. AdaBoost
🧠 Idea

👉 Focus on mistakes

Each new model:

pays more attention to wrong predictions
Use
Improve weak models
⚠️ When to use
Moderate datasets
⚡ 5. XGBoost
🧠 Idea

👉 Advanced boosting (fast + powerful)

Use
Competitions
High accuracy tasks
🔥 Why popular?
Speed + performance
🔴 6. Perceptron (Single & Multilayer)
🧠 Idea

👉 Basic neural network

Single:

simple linear decision

Multi-layer:

can learn complex patterns
Use
Deep learning basics
Example

Digit recognition

🟡 7. K-Means
🧠 Idea

👉 Group similar data

Use
Customer segmentation
Clustering
Example

Group users into spending categories

🟠 8. K-Modes
🧠 Idea

👉 Like K-Means but for categorical data

Example
Gender, city, job type
⚠️ When to use
Data is NOT numeric
🟣 9. PCA (Principal Component Analysis)
🧠 Idea

👉 Reduce number of features

Use
Speed up model
Visualization
Example

100 features → 2 features

🟢 10. SOM (Self Organizing Map)
🧠 Idea

👉 Map high-dim → 2D grid

Use
Visualization
Pattern discovery
🎮 11. Q-Learning
🧠 Idea

👉 Learn via rewards

Agent learns:

best action in each state
Example
Game AI
Robot navigation























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
