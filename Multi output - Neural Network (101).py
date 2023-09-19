#!/usr/bin/env python
# coding: utf-8

# ## Developing a Multi-Output Neural Network 

# In[1]:


import pandas as pd
import os
desktop_path = os.path.expanduser("~/Desktop")
file_path = os.path.join(desktop_path, "random_subset.xlsx")
df = pd.read_excel(file_path)
print(df.head())


# In[2]:


import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model


# In[3]:


reviews = df['cleaned_review'].tolist()
sentiments = df['sentiment'].tolist()
categories = df['predicted_category'].tolist()


# In[4]:


sentiment_mapping = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
category_mapping = {'Care': 0, 'General': 1, 'Leads': 2}


# In[5]:


sentiment_labels = [sentiment_mapping[label] for label in sentiments]
category_labels = [category_mapping[label] for label in categories]


# In[6]:


max_words = 10000  # Choose an appropriate vocabulary size
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')


# In[7]:


train_reviews, val_reviews, train_sentiments, val_sentiments, train_categories, val_categories = train_test_split(
    padded_sequences, sentiment_labels, category_labels, test_size=0.2, random_state=42
)


# In[8]:


# Define the model
input_layer = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(input_dim=max_words, output_dim=128)(input_layer)
lstm_layer = LSTM(64)(embedding_layer)


# In[9]:


# Separate output layers for sentiment and category
num_sentiments = len(sentiment_mapping)
num_categories = len(category_mapping)


# In[10]:


sentiment_output = Dense(num_sentiments, activation='softmax', name='sentiment')(lstm_layer)
category_output = Dense(num_categories, activation='softmax', name='category')(lstm_layer)


# In[11]:


model = Model(inputs=input_layer, outputs=[sentiment_output, category_output])


# In[12]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[13]:


train_sentiments = np.array(train_sentiments)
train_categories = np.array(train_categories)
val_sentiments = np.array(val_sentiments)
val_categories = np.array(val_categories)


# In[21]:


# Train the model
model.fit(train_reviews, [train_sentiments, train_categories], epochs=10, batch_size=32,
          validation_data=(val_reviews, [val_sentiments, val_categories]))


# In[67]:


import matplotlib.pyplot as plt


# In[71]:


log_data = {
    'loss': [2.1093, 1.9209, 1.6669, 1.3628, 1.3186, 1.2684, 1.2071, 1.1862, 1.1915, 1.1904],
    'val_loss': [2.0080, 1.8089, 1.5602, 1.6116, 1.5998, 1.4914, 1.4283, 1.4007, 1.3870, 1.3844],
    'accuracy': [0.8250, 0.8250, 0.8250, 0.8250, 0.8250, 0.8250, 0.8250, 0.8250, 0.8250, 0.8250],
    'val_accuracy': [0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000]
}

# Number of epochs
epochs = range(1, len(log_data['loss']) + 1)


# In[72]:


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, log_data['loss'], label='Training Loss')
plt.plot(epochs, log_data['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# In[73]:


plt.subplot(1, 2, 2)
plt.plot(epochs, log_data['accuracy'], label='Training Accuracy')
plt.plot(epochs, log_data['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# In[22]:


# Save the model
model.save('multi_output_model.h5')
print("Model saved successfully")


# ## Validation

# In[14]:


excel_path = '/home/xjoex/Desktop/review_text.xlsx'
df_loaded = pd.read_excel(excel_path)
print(df_loaded.head())


# In[15]:


# Preprocess text data from df_loaded
reviews_subset = df_loaded['cleaned_review'].tolist()[:200] 
sequences_subset = tokenizer.texts_to_sequences(reviews_subset)
padded_sequences_subset = pad_sequences(sequences_subset, maxlen=max_sequence_length, padding='post')


# In[16]:


sentiment_probs, category_probs = model.predict(padded_sequences_subset)


# In[17]:


# Convert probabilities to labels
predicted_sentiments = np.argmax(sentiment_probs, axis=1)
predicted_categories = np.argmax(category_probs, axis=1)


# In[18]:


reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
reverse_category_mapping = {v: k for k, v in category_mapping.items()}


# In[19]:


predicted_sentiment_labels = [reverse_sentiment_mapping[sentiment] for sentiment in predicted_sentiments]
predicted_category_labels = [reverse_category_mapping[category] for category in predicted_categories]


# In[20]:


predictions_df = pd.DataFrame({
    'Review': reviews_subset,
    'Predicted_Sentiment': predicted_sentiment_labels,
    'Predicted_Category': predicted_category_labels
})


# In[21]:


predictions_df.head(10)


# In[ ]:


desktop_path = f"/home/xjoex/Desktop"  
predictions_excel_filename = "predictions.xlsx"
predictions_excel_path = f"{desktop_path}/{predictions_excel_filename}"

predictions_df.to_excel(predictions_excel_path, index=False)

print(f"Predictions saved to: {predictions_excel_path}")


# ## Developing a Semi-Supervised Neural Network

# In[22]:


import pandas as pd
file_path = '/home/xjoex/Desktop/review_text.xlsx'
df_review_text = pd.read_excel(file_path)
print(df_review_text.head())


# In[23]:


unlabeled_reviews = df_review_text['cleaned_review'].astype(str).tolist()
unlabeled_sequences = tokenizer.texts_to_sequences(unlabeled_reviews)
padded_unlabeled_sequences = pad_sequences(unlabeled_sequences, maxlen=max_sequence_length, padding='post')


# In[24]:


# Pseudo-label the unlabeled data
unlabeled_sentiments_probs, unlabeled_categories_probs = model.predict(padded_unlabeled_sequences)
pseudo_sentiments = np.argmax(unlabeled_sentiments_probs, axis=1)
pseudo_categories = np.argmax(unlabeled_categories_probs, axis=1)


# In[ ]:


pseudo_labeled_data = pd.DataFrame({
    'cleaned_review': unlabeled_reviews,
    'pseudo_sentiment': pseudo_sentiments,
    'pseudo_category': pseudo_categories
})


# In[ ]:


desktop_path = os.path.expanduser("~/Desktop")


# In[ ]:


excel_path = os.path.join(desktop_path, "pseudo_labeled_data_full.xlsx")
pseudo_labeled_data.to_excel(excel_path, index=False)
print("Pseudo-labeled data saved to Excel on the desktop successfully.")


# In[ ]:


# Combine labeled and pseudo-labeled data
combined_reviews = np.concatenate((train_reviews, padded_unlabeled_sequences[:6000]))
combined_sentiments = np.concatenate((train_sentiments, pseudo_sentiments[:6000]))
combined_categories = np.concatenate((train_categories, pseudo_categories[:6000]))


# In[39]:


# Train the model with combined data
model.fit(combined_reviews, [combined_sentiments, combined_categories], epochs=10, batch_size=32,
          validation_data=(val_reviews, [val_sentiments, val_categories]))


# In[74]:


log_data = {
    'loss': [0.0592, 0.0403, 0.0404, 0.0401, 0.0401, 0.0404, 0.0400, 0.0402, 0.0403, 0.0401],
    'val_loss': [3.5371, 3.5121, 3.4244, 3.5458, 3.6220, 3.4174, 3.7107, 3.5053, 3.5374, 3.7344],
    'accuracy': [0.9977, 0.9977, 0.9977, 0.9977, 0.9977, 0.9977, 0.9977, 0.9977, 0.9977, 0.9977],
    'val_accuracy': [0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000]
}


# In[75]:


epochs = range(1, len(log_data['loss']) + 1)


# In[76]:


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, log_data['loss'], label='Training Loss')
plt.plot(epochs, log_data['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# In[77]:


plt.subplot(1, 2, 2)
plt.plot(epochs, log_data['accuracy'], label='Training Accuracy')
plt.plot(epochs, log_data['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# In[40]:


model.save('trained_multi_output_modelIT4.h5')
print("Trained model saved successfully")


# #### Evaluating the Model 

# In[25]:


from keras.models import load_model


# In[26]:


loaded_model = load_model('trained_multi_output_modelIT4.h5')


# In[27]:


random_subset = df_review_text.sample(n=200, random_state=42)
random_subset_reviews = random_subset['cleaned_review'].tolist()
random_subset_sequences = tokenizer.texts_to_sequences(random_subset_reviews)
padded_random_subset_sequences = pad_sequences(random_subset_sequences, maxlen=max_sequence_length, padding='post')


# In[28]:


predicted_sentiments_probs, predicted_categories_probs = loaded_model.predict(padded_random_subset_sequences)
predicted_sentiments = [reverse_sentiment_mapping[np.argmax(probs)] for probs in predicted_sentiments_probs]
predicted_categories = [reverse_category_mapping[np.argmax(probs)] for probs in predicted_categories_probs]


# In[29]:


random_subset['predicted_sentiment'] = predicted_sentiments
random_subset['predicted_category'] = predicted_categories


# In[30]:


print(random_subset)


# In[31]:


random_subset.head(10)


# In[32]:


true_sentiments = random_subset['predicted_sentiment']
true_categories = random_subset['predicted_category']


# In[33]:


sentiment_accuracy = (predicted_sentiments == true_sentiments).mean()
category_accuracy = (predicted_categories == true_categories).mean()


# ### Hyperparameter Tuning 

# In[34]:


from itertools import product
import numpy as np
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model


# In[35]:


# Define hyperparameter values for tuning
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]
num_lstm_units = [64, 128, 256]
embedding_dims = [50, 100, 200]
dropout_rates = [0.2, 0.3, 0.4]


# In[36]:


best_accuracy = 0.0
best_hyperparameters = {}
best_model = None


# In[37]:


for lr, batch_size, lstm_units, embedding_dim, dropout_rate in product(learning_rates, batch_sizes, num_lstm_units, embedding_dims, dropout_rates):
    input_layer = Input(shape=(max_sequence_length,))
    embedding_layer = Embedding(input_dim=max_words, output_dim=embedding_dim)(input_layer)
    lstm_layer = LSTM(lstm_units)(embedding_layer)
    
    sentiment_output = Dense(num_sentiments, activation='softmax', name='sentiment')(lstm_layer)
    category_output = Dense(num_categories, activation='softmax', name='category')(lstm_layer)
    
    model = Model(inputs=input_layer, outputs=[sentiment_output, category_output])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[70]:


model.fit(train_reviews, [train_sentiments, train_categories], epochs=10, batch_size=batch_size, verbose=0)


# In[38]:


val_accuracy = np.mean(model.evaluate(val_reviews, [val_sentiments, val_categories], verbose=0)[1])


# In[39]:


# Check if this model's accuracy is better than the current best
if val_accuracy > best_accuracy:
    best_accuracy = val_accuracy
    best_hyperparameters = {
        'learning_rate': lr,
        'batch_size': batch_size,
        'num_lstm_units': lstm_units,
        'embedding_dim': embedding_dim,
        'dropout_rate': dropout_rate
    }
    best_model = model


# In[40]:


# Print the best hyperparameters and accuracy
print("Best Hyperparameters:")
print(best_hyperparameters)
print("Best Validation Accuracy:", best_accuracy)


# In[75]:


best_model.save('tuned_model.h5')


# In[76]:


#tuned_model = load_model('tuned_model.h5')


# In[41]:


from sklearn.model_selection import StratifiedKFold  
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  


# In[42]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 


# ### Validation from Tuned Model

# In[43]:


tuned_model = load_model('tuned_model.h5')


# In[44]:


random_subset.head(10)


# In[45]:


subset_sequences = tokenizer.texts_to_sequences(subset_reviews)
padded_subset_sequences = pad_sequences(subset_sequences, maxlen=max_sequence_length, padding='post')


# In[46]:


sentiment_probs, category_probs = tuned_model.predict(padded_subset_sequences)


# In[47]:


predicted_sentiments = [reverse_sentiment_mapping[np.argmax(probs)] for probs in sentiment_probs]
predicted_categories = [reverse_category_mapping[np.argmax(probs)] for probs in category_probs]


# In[48]:


predictions_df = pd.DataFrame({
    'Review': subset_reviews,
    'Predicted_Sentiment': predicted_sentiments,
    'Predicted_Category': predicted_categories
})


# In[49]:


num_rows = random_subset.shape[0]
print("Number of Rows in random_subset:", num_rows)


# In[104]:


random_subset.to_excel('predictions773.xlsx', index=False)


# ### Evaluating the model 

# In[50]:


excel_file_path = '/home/xjoex/Downloads/predictions100.xlsx'


# In[51]:


data = pd.read_excel(excel_file_path)


# In[52]:


data.head(10)


# In[53]:


import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# In[54]:


sentiment_accuracy = accuracy_score(data['Ground Truth_Sentiment'], data['predicted_sentiment'])


# In[55]:


sentiment_precision = precision_score(data['Ground Truth_Sentiment'], data['predicted_sentiment'], average='weighted')


# In[56]:


sentiment_recall = recall_score(data['Ground Truth_Sentiment'], data['predicted_sentiment'], average='weighted')


# In[57]:


sentiment_f1_score = f1_score(data['Ground Truth_Sentiment'], data['predicted_sentiment'], average='weighted')


# In[58]:


category_accuracy = accuracy_score(data['Ground Truth_category'], data['predicted_category'])


# In[59]:


category_precision = precision_score(data['Ground Truth_category'], data['predicted_category'], average='weighted')


# In[60]:


category_recall = recall_score(data['Ground Truth_category'], data['predicted_category'], average='weighted')


# In[61]:


category_f1_score = f1_score(data['Ground Truth_category'], data['predicted_category'], average='weighted')


# In[62]:


# Calculate the loss
incorrect_sentiment_predictions = (data['Ground Truth_Sentiment'] != data['predicted_sentiment']).sum()
incorrect_category_predictions = (data['Ground Truth_category'] != data['predicted_category']).sum()
total_predictions = len(data)


# In[63]:


sentiment_loss = incorrect_sentiment_predictions / total_predictions
category_loss = incorrect_category_predictions / total_predictions


# In[64]:


# Print the calculated metrics
print("Sentiment Accuracy:", sentiment_accuracy)
print("Category Accuracy:", category_accuracy)
print("Sentiment Precision:", sentiment_precision)
print("Sentiment Recall:", sentiment_recall)
print("Sentiment F1 Score:", sentiment_f1_score)
print("Category Precision:", category_precision)
print("Category Recall:", category_recall)
print("Category F1 Score:", category_f1_score)
print("Sentiment Loss:", sentiment_loss)
print("Category Loss:", category_loss)


# In[65]:


sentiment_accuracy_percent = sentiment_accuracy * 100
category_accuracy_percent = category_accuracy * 100
sentiment_precision_percent = sentiment_precision * 100
sentiment_recall_percent = sentiment_recall * 100
sentiment_f1_score_percent = sentiment_f1_score * 100
category_precision_percent = category_precision * 100
category_recall_percent = category_recall * 100
category_f1_score_percent = category_f1_score * 100
sentiment_loss_percent = (incorrect_sentiment_predictions / total_predictions) * 100
category_loss_percent = (incorrect_category_predictions / total_predictions) * 100

print("Sentiment Accuracy (%):", sentiment_accuracy_percent)
print("Category Accuracy (%):", category_accuracy_percent)
print("Sentiment Precision (%):", sentiment_precision_percent)
print("Sentiment Recall (%):", sentiment_recall_percent)
print("Sentiment F1 Score (%):", sentiment_f1_score_percent)
print("Category Precision (%):", category_precision_percent)
print("Category Recall (%):", category_recall_percent)
print("Category F1 Score (%):", category_f1_score_percent)
print("Sentiment Loss (%):", sentiment_loss_percent)
print("Category Loss (%):", category_loss_percent)


# In[78]:


hyperparameters = {
    'Learning Rate': 0.1,
    'Batch Size': 64,
    'Number of LSTM Units': 256,
    'Embedding Dimension': 200,
    'Dropout Rate': 0.4
}


# In[79]:


metrics = {
    'Sentiment Accuracy (%)': 89.8989898989899,
    'Category Accuracy (%)': 88.88888888888889,
    'Sentiment Precision (%)': 89.8989898989899,
    'Sentiment Recall (%)': 89.8989898989899,
    'Sentiment F1 Score (%)': 89.8989898989899,
    'Category Precision (%)': 89.23057737872553,
    'Category Recall (%)': 88.88888888888889,
    'Category F1 Score (%)': 89.04087696467418,
    'Sentiment Loss (%)': 10.1010101010101,
    'Category Loss (%)': 11.11111111111111
}


# In[81]:


fig, axs = plt.subplots(1, 2, figsize=(15, 6))

axs[0].bar(hyperparameters.keys(), hyperparameters.values(), color='skyblue')
axs[0].set_title('Hyperparameters')
axs[0].set_ylabel('Values')

axs[1].bar(metrics.keys(), metrics.values(), color='lightcoral')
axs[1].set_title('Evaluation Metrics')
axs[1].set_ylabel('Values')

for ax in axs:
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# In[82]:


import plotly.express as px
import pandas as pd

hyperparameters_df = pd.DataFrame.from_dict(hyperparameters, orient='index', columns=['Value'])
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])

fig = px.bar(hyperparameters_df, x=hyperparameters_df.index, y='Value', title='Hyperparameters', labels={'Value': 'Values'})
fig2 = px.bar(metrics_df, x=metrics_df.index, y='Value', title='Evaluation Metrics', labels={'Value': 'Values'})

fig.update_xaxes(tickangle=45)
fig2.update_xaxes(tickangle=45)

fig.add_trace(fig2.data[0])

fig.show()


# In[ ]:




