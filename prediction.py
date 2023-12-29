import mlflow.pyfunc
import pandas as pd

# Load the deployed model
model = mlflow.pyfunc.load_model("file:/home/dhruba/.config/zenml/local_stores/b7a1fdb7-74f5-49c0-990b-5a51d202b3c2/mlruns")

# Prepare input data
input_data = pd.DataFrame({
    'feature1': [value1],
    'feature2': [value2],
    # Add other features as needed
})

# Make predictions
predictions = model.predict(input_data)

print(predictions)
