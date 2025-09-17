from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("winequality-red.csv", delimiter = ";")

fig = plt.figure()

#mplot3d is needed to set projection='3d'
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['fixed acidity'], df['residual sugar'], df['density'], c='b', marker='o')

ax.set_xlabel('volatile acidity, X1')
ax.set_ylabel('free sulfur dioxide, X2')
ax.set_zlabel('density, Y')

plt.show()