#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO

# For downloading images
import os
import urllib


# In[2]:


#bot_keyboard
import pandas as pd
column_names = ["BKEYTime", "BKEYAction", "BKEYX_Coordinate", "BKEYY_Coordinate"]
bot_keyboard = pd.read_csv(r"bot_keyboard.txt", sep=",", names=column_names, skiprows=1)
print(bot_keyboard.head())


# In[3]:


#bot_naturalmouse
import pandas as pd
column_names1 = ["BMOUSETime", "BMOUSEAction", "BMOUSEX_Coordinate", "BMOUSEY_Coordinate"]
bot_naturalmouse = pd.read_csv(r"bot_naturalmouse.txt", sep=",", names=column_names1, skiprows=1)
print(bot_naturalmouse.head())


# In[4]:


#human_vm
import pandas as pd
column_names2 = ["HTime", "HAction", "HX_Coordinate", "HY_Coordinate"]
human_vm = pd.read_csv(r"humanpv.txt", sep=",", names=column_names2, skiprows=1)
print(human_vm.head())


# In[5]:


bot_keyboard['label'] = 0
bot_naturalmouse['label'] = 0
human_vm['label'] = 1


# In[6]:


import pandas as pd
column_names = ["BKEYTime", "BKEYAction", "BKEYX_Coordinate", "BKEYY_Coordinate"]
bot_keyboard = pd.read_csv(r"bot_keyboard.txt", sep=",", names=column_names, skiprows=1)
column_names1 = ["BMOUSETime", "BMOUSEAction", "BMOUSEX_Coordinate", "BMOUSEY_Coordinate"]
bot_naturalmouse = pd.read_csv(r"bot_naturalmouse.txt", sep=",", names=column_names1, skiprows=1)
column_names2 = ["HTime", "HAction", "HX_Coordinate", "HY_Coordinate"]
human_vm = pd.read_csv(r"humanpv.txt", sep=",", names=column_names2, skiprows=1)

bot_keyboard.reset_index(drop=True, inplace=True)
bot_naturalmouse.reset_index(drop=True, inplace=True)
human_vm.reset_index(drop=True, inplace=True)

combined_df = pd.concat([bot_keyboard, bot_naturalmouse, human_vm], axis=1)
print(combined_df.head())


# In[7]:


combined_df['BKEYAction'].value_counts()


# In[8]:


combined_df['BMOUSEAction'].value_counts()


# In[9]:


combined_df['HAction'].value_counts()


# In[10]:


len_bot_keyboard = len(bot_keyboard)
len_bot_naturalmouse = len(bot_naturalmouse)
len_human_vm = len(human_vm)

print(f"Length of bot_keyboard: {len_bot_keyboard}")
print(f"Length of bot_naturalmouse: {len_bot_naturalmouse}")
print(f"Length of human_vm: {len_human_vm}")



# In[11]:


# Truncate the datasets to the length of combined_df
length_combined = len(combined_df)
bot_naturalmouse = bot_naturalmouse.iloc[:length_combined]


# In[12]:


if len(bot_keyboard) < length_combined:
    # Interpolate or repeat rows
    bot_keyboard = bot_keyboard.reindex(range(length_combined), method='ffill')  # Forward fill

# For human_vm (shorter)
if len(human_vm) < length_combined:
    # Interpolate or repeat rows
    human_vm = human_vm.reindex(range(length_combined), method='ffill')  # Forward fill


# In[26]:


# Assuming bot_naturalmouse already has the correct length
combined_df = pd.concat([bot_keyboard, bot_naturalmouse, human_vm], axis=1)


# In[27]:


# Create labels
label_bot_keyboard = [0] * len(bot_keyboard)
label_bot_naturalmouse = [0] * len(bot_naturalmouse)
label_human_vm = [1] * len(human_vm)

# Combine labels
labels = label_bot_keyboard + label_bot_naturalmouse + label_human_vm
labels = labels[:length_combined]  # Truncate or extend to match length_combined

# Add labels to DataFrame
combined_df['label'] = labels


# In[30]:


from sklearn.preprocessing import LabelEncoder

# Define features and target variable
X = combined_df.drop(columns=['label'])
y = combined_df['label']

# Convert non-numeric columns to numeric
categorical_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in categorical_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# Check for and handle missing values
X = X.ffill().bfill()

# Ensure y is numeric
y = y.astype(int)


# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[32]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Bot', 'Human'], yticklabels=['Bot', 'Human'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nFeature Ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. feature {X.columns[indices[f]]} ({importances[indices[f]]:.4f})")


# In[ ]:




