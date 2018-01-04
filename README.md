Test


```python
# Train a pool of 10 classifiers
model = CalibratedClassifierCV(Perceptron())   
pool_classifiers = BaggingClassifier(model, n_estimators=10)
pool_classifiers.fit(X_train, y_train)

# Initialize the DES model
knorae = KNORAE(pool_classifiers)

# Preprocess the DSEL dataset
knorae.fit(X_dsel, y_dsel)

# Predict new examples:
knorae.predict(X_test)

```
