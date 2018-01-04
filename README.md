Test

Example using the KNORA-E method with random forest as a pool of classifiers:

```python
from sklearn.ensemble import RandomForestClassifier
from pythonds.des.knora_e import KNORAE


# Train a pool of 10 classifiers
pool_classifiers = RandomForestClassifier(n_estimators=10)
pool_classifiers.fit(X_train, y_train)

# Initialize the DES model
knorae = KNORAE(pool_classifiers)

# Preprocess the Dynamic Selection dataset (DSEL)
knorae.fit(X_dsel, y_dsel)

# Predict new examples:
knorae.predict(X_test)

```
