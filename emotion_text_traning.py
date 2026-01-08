


import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pickle
import json
import glob

class EmotionText():
    def __init__(self, dataset_path=None):
        self.model = None
        self.labels = ["anger", "fear", "joy", "love", "sadness", "surprise"]  # 6 labels
        self.dataset_path = dataset_path
        self.tokenizer = None
        self.history = None
        self.num_classes = 6
    
    def load_data(self, file_path=None):
        
        if file_path is None:
            file_path = self.dataset_path
        
        
        if file_path and os.path.isdir(file_path):
            csv_files = glob.glob(os.path.join(file_path, '*.csv'))
            if csv_files:
                file_path = csv_files[0]
                print(f"Found CSV file in directory: {file_path}")
            else:
                
                all_csv_files = glob.glob('/kaggle/input/**/*.csv', recursive=True)
                if all_csv_files:
                    file_path = all_csv_files[0]
                    print(f"Found CSV file: {file_path}")
                else:
                    raise FileNotFoundError("No CSV files found")
        elif not file_path:
            
            all_csv_files = glob.glob('/kaggle/input/**/*.csv', recursive=True)
            if all_csv_files:
                file_path = all_csv_files[0]
                print(f"Found CSV file: {file_path}")
            else:
                raise ValueError("Please provide dataset path")
        
        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            raise FileNotFoundError(f"CSV file not found. Error: {e}")
        
     
        text_column = 'text'
        label_column = 'label'
        
        if 'text' not in data.columns:
            text_candidates = [col for col in data.columns if 'text' in col.lower() or 'sentence' in col.lower()]
            if text_candidates:
                text_column = text_candidates[0]
            else:
                text_column = data.columns[0]
        
        if 'label' not in data.columns:
            label_candidates = [col for col in data.columns if 'label' in col.lower() or 'emotion' in col.lower()]
            if label_candidates:
                label_column = label_candidates[0]
            else:
                label_column = data.columns[1]
        
        x = data[text_column].astype(str).tolist()
        y = data[label_column].values
        
        print(f"Dataset loaded: {len(x)} samples")
        print(f"Text column: '{text_column}', Label column: '{label_column}'")
        print(f"Classes found: {set(y)}")
        
        return x, y
    
    def text_number(self, file_path=None):
       
        x, y = self.load_data(file_path)
        
        
        print(f"y shape: {y.shape}")
        print(f"y sample: {y[:5]}")
        
      
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=75000, 
            oov_token='<OOV>'
        )
        self.tokenizer.fit_on_texts(x)
        
       
        x_seq = self.tokenizer.texts_to_sequences(x)
        
       
        x_padded = tf.keras.preprocessing.sequence.pad_sequences(
            x_seq, 
            maxlen=100, 
            padding='post', 
            truncating='post'
        )
        
        
        self.save_tokenizer()
        
        print(f"Text processing complete. Vocabulary size: {len(self.tokenizer.word_index)}")
        return x_padded, y
    
    def save_tokenizer(self):
        
        if self.tokenizer:
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Tokenizer saved as tokenizer.pickle")
    
    def load_tokenizer(self):
        
        try:
            with open('tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            print("Tokenizer loaded from tokenizer.pickle")
        except:
            print("No saved tokenizer found")
            self.tokenizer = None
    
    def build_model(self):
        
        from tensorflow.keras import layers, models
        
        inp = layers.Input(shape=(100))
        
        # Embedding layer
        x = layers.Embedding(
            input_dim=50000, 
            output_dim=128, 
            input_length=100,
            mask_zero=True
        )(inp)
        
        # MultiHeadAttention
        x = layers.MultiHeadAttention(num_heads=4, key_dim=128)(x, x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Reshape for LSTM
        x = layers.Reshape((1, -1))(x)
        
        # Bidirectional LSTM layers
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(32))(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        out = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs=inp, outputs=out)
        
       
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        self.model.summary()
        print("Model built successfully!")
    
    def train_model(self, file_path=None, epochs=3, batch_size=32):
        
        x, y = self.text_number(file_path)
        
       
        if len(y.shape) == 1:
            # Integer labels to one-hot
            y = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
            print("Converted integer labels to one-hot encoding")
        elif y.shape[1] != self.num_classes:
            # Convert to one-hot
            y = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
            print("Converted labels to one-hot encoding")
        
       
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1) if len(y.shape) > 1 else y
        )
        
        print(f"Training samples: {len(x_train)}")
        print(f"Validation samples: {len(x_val)}")
        
        
        self.build_model()
        
        
        os.makedirs('checkpoints', exist_ok=True)
        
        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=1
        )
        
      
        checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
      
        checkpoint_epoch = tf.keras.callbacks.ModelCheckpoint(
            'model_epoch_{epoch:02d}.h5',
            save_freq='epoch',
            verbose=0
        )
        
        callbacks = [
            early_stopping,
            reduce_lr,
            checkpoint_best,
            checkpoint_epoch
        ]
        
        
        self.history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
       
        self.model.save('checkpoints/final_model.h5')
        print("Training completed!")
        
        return self.history
    
    def load_model(self, model_path='checkpoints/best_model.h5'):
       
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
            return True
        else:
            print(f"Model not found at {model_path}")
            return False
    
    def evaluate_model(self, file_path=None):
        
        if self.model is None:
            print("Model not loaded. Loading best model")
            self.load_model()
        
        x, y = self.text_number(file_path)
        
        if len(y.shape) == 1:
            y = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
        
       
        _, x_test, _, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        
        
        y_pred_probs = self.model.predict(x_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix1.png')
        plt.show()
        
       
        accuracy = np.mean(y_pred == y_true)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        return y_true, y_pred
    
    def visualize_training(self):
        
        if self.history is None:
            print("No training history available")
            return
        
        history = self.history.history
        
        plt.figure(figsize=(12, 4))
        
       
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        
       
        history_df = pd.DataFrame(history)
        history_df.to_csv('training_history.csv', index=False)
        print("Training history saved to training_history.csv")
    
    def predict(self, text):
        
        if self.model is None:
            print("Model not loaded. Loading best model")
            self.load_model()
        
        if self.tokenizer is None:
            self.load_tokenizer()
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available. Please train model first.")
        
      
        seq = self.tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            seq, 
            maxlen=100, 
            padding='post', 
            truncating='post'
        )
        
        
        pred = self.model.predict(padded, verbose=0)
        pred_class = np.argmax(pred)
        confidence = np.max(pred)
        
       
        if pred_class < len(self.labels):
            emotion = self.labels[pred_class]
        else:
            emotion = f"Class_{pred_class}"
        
        return emotion, confidence, pred[0]


if __name__ == "__main__":
    emotion_model = EmotionText()
    history = emotion_model.train_model(epochs=5)
    emotion_model.evaluate_model()
    emotion_model.visualize_training()
  