import requests
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

# Example dataset
# Replace this with actual data and labels
# Example format: [temperature, wind_speed, humidity, pressure]
# Labels: 0 = no disaster, 1 = disaster
data = np.array([
    [25, 10, 80, 1015, 0],
    [30, 15, 70, 1020, 1],
    [20, 5, 85, 1005, 0],
    # Add more data here
])
X = data[:, :-1]  # Features: temperature, wind_speed, humidity, pressure
y = data[:, -1]  # Target: disaster_occurred

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)

# Create the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Weather data from OpenWeatherMap API
baseURL = "https://api.openweathermap.org/data/2.5/weather?q="
apikey = "91abe413fd6e8c031c8a97a0523e70e6"
city = input("Enter name of the city: ")
completeURL = baseURL + city + "&appid=" + apikey
response = requests.get(completeURL)
data = response.json()

# Check if the API response contains valid data
if response.status_code == 200 and "main" in data and "wind" in data:
    # Extract weather information
    a = data["main"]["temp"] - 273.15  # Convert from Kelvin to Celsius
    b = data["wind"]["speed"]
    c = data["main"]["humidity"]
    d = data["main"]["pressure"]

    # Prepare new data for prediction
    new_data = np.array([[a, b, c, d]])

    # Predict disaster
    disaster_prediction = clf.predict(new_data)

    if disaster_prediction[0] == 1:
        print("Disaster predicted.")
    else:
        print("No disaster predicted.")