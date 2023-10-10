from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns


# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values: ")
    print(y_pred)
    return y_pred


df = sns.load_dataset('iris')
df.info()
df.isnull().any()
df.shape
target = df['species']
df1 = df.copy()
df1 = df1.drop('species', axis=1)
df1.shape
df1.head()
# Defining the attributes
X = df1
target
# label encoding
le = LabelEncoder()
target = le.fit_transform(target)
target
y = target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)

print("Training split input: ", X_train.shape)
print("Testing split input: ", X_test.shape)

# Creating Decision Tree Classifier
dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(X_train, y_train)
gini_dtree = DecisionTreeClassifier(criterion='gini')
gini_dtree.fit(X_train, y_train)

# accuracy
# entropy
y_pred = prediction(X_test, dtree)
entropy_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy using entopry: ", entropy_accuracy)

# gini
y_pred_gini = prediction(X_test, gini_dtree)
gini_accuracy = accuracy_score(y_test, y_pred_gini)
print("Accuracy using gini: ", gini_accuracy)

if entropy_accuracy > gini_accuracy:
    print("\nAccuracy with entropy is higher\n")
else:
    print("\nAccuracy with gini index is higher\n")

# Decision Tree plotting
plt.figure(figsize=(20, 20))
dec_tree = plot_tree(decision_tree=dtree, feature_names=df1.columns,
                     class_names=["setosa", "vercicolor", "verginica"], filled=True, precision=4, rounded=True)

plt.savefig("IrisTree_Entropy.png")

plt.figure(figsize=(20, 20))
gini_dec_tree = plot_tree(decision_tree=gini_dtree, feature_names=df1.columns,
                          class_names=["setosa", "vercicolor", "verginica"], filled=True, precision=4, rounded=True)

plt.savefig("IrisTree_Gini.png")

plt.show()
