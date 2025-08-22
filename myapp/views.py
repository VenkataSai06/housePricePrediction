# myapp/views.py

from django.shortcuts import render
from django.http import JsonResponse
import joblib
import os
import numpy as np
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import json

# Define the full path to the saved model
MODEL_PATH = os.path.join(settings.BASE_DIR, 'model', 'house_price_model.joblib')

# Load the model only once when the application starts
# This is a key optimization to prevent re-loading the model for every request
try:
    model = joblib.load(MODEL_PATH)
    print("Machine learning model loaded successfully.")
except FileNotFoundError:
    model = None
    print(f"Error: Model file not found at {MODEL_PATH}. Please run train_model.py first.")

def index(request):
    """
    Renders the main page with the prediction form.
    """
    return render(request, 'index.html')

@csrf_exempt
def predict(request):
    """
    Handles the prediction request.
    """
    if request.method == 'POST':
        # Check if the model was loaded successfully
        if not model:
            return JsonResponse({'error': 'Model not found. Please contact the administrator.'}, status=500)
            
        try:
            # Parse the incoming JSON data from the form submission
            data = json.loads(request.body)
            
            # Extract the features
            area = data.get('area')
            bedrooms = data.get('bedrooms')
            bathrooms = data.get('bathrooms')
            
            # Validate the input data
            if not all(isinstance(val, (int, float)) for val in [area, bedrooms, bathrooms]):
                return JsonResponse({'error': 'Invalid input data. Please provide numbers.'}, status=400)
            
            # Prepare the input for the model
            input_data = np.array([[area, bedrooms, bathrooms]])
            
            # Make a prediction
            prediction = model.predict(input_data)[0]
            
            # Format the prediction to a readable string (e.g., "$500,000.00")
            predicted_price = f"${prediction:,.2f}"
            
            # Return the prediction as a JSON response
            return JsonResponse({'prediction': predicted_price, 'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms})
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON format.'}, status=400)
        except Exception as e:
            # Catch any other potential errors and return a server error
            return JsonResponse({'error': str(e)}, status=500)

    # For any other request method, return an error
    return JsonResponse({'error': 'Invalid request method.'}, status=405)
