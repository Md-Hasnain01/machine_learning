from joblib import load
import numpy as np

# from redwine import model

# load(model, 'redwine.joblib')
model = load('redwine.joblib')
input_the_data = np.array([[0.0148335, 0.46835924, -0.26769975, 0.05870144, -0.01010418, -0.94818008,
                            -0.83046193, 0.82139562, -0.33359616, -0.28384054, -1.05550392]])

predictions = model.predict(input_the_data)
print("Quality =", predictions)