# model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
import joblib
from sklearn.preprocessing import LabelEncoder
from privacy_preserving import preprocess_data_with_privacy
import scipy.sparse


le = LabelEncoder()
# Load dataset
data = pd.read_csv('phishing_dataset.csv')
data = data.dropna()
# Preprocess dataset
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['URL'])
y = data['Label']
y = le.fit_transform(y)

# Apply privacy-preserving preprocessing
X_privacy = preprocess_data_with_privacy(X.toarray(), epsilon=0.1)

# Perform Random Oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_privacy, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save the model and vectorizer
joblib.dump(model, 'phishing_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
