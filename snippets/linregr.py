import numpy as np
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

X = np.random.randn(6, 2)

y = np.array([0, 1, 2, 1, 2, 0])

model.fit(X, y)

print('Linear Regression Classifier: y = W * X + b')

for i, x_ in enumerate(X):
    print('(%.4f, %.4f) ->' % (x_[0], x_[1]), y[i])

print('W =', model.coef_)
print('b =', model.intercept_)

x_test = np.array([0.5, 0.5]).reshape(1, -1)
print('(%.4f, %.4f) ->' % (x_test[0][0], x_test[0][1]),
      model.predict_proba(x_test))
