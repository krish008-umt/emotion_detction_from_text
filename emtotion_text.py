
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import glob

class EmotionTextPredictor():
    def __init__(self, model_path=r"C:\Users\Dell\Downloads\best_model.h5", tokenizer_path=r"C:\Users\Dell\Downloads\tokenizer.pickle"):
        self.model = None
        self.labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]  # 6 labels
        self.tokenizer = None
        self.num_classes = 6
        
       
        self.load_model(model_path)
        self.load_tokenizer(tokenizer_path)
    
    def load_model(self, model_path='best_model.h5'):
        
        try:
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f" Model loaded from {model_path}")
                print(f"   Model summary:")
                self.model.summary()
                return True
            else:
                print(f" Model not found at {model_path}")
                print(f"   Looking for model in current directory...")
               
                model_files = glob.glob('*.h5') + glob.glob('checkpoints/*.h5')
                if model_files:
                    model_path = model_files[0]
                    self.model = tf.keras.models.load_model(model_path)
                    print(f" Model loaded from {model_path}")
                    return True
                else:
                    print(f" No model files found. Please train a model first.")
                    return False
        except Exception as e:
            print(f" Error loading model: {e}")
            return False
    
    def load_tokenizer(self, tokenizer_path='tokenizer.pickle'):
        
        try:
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as handle:
                    self.tokenizer = pickle.load(handle)
                print(f" Tokenizer loaded from {tokenizer_path}")
                print(f"   Vocabulary size: {len(self.tokenizer.word_index)}")
                return True
            else:
                print(f" Tokenizer not found at {tokenizer_path}")
                print(f"   Looking for tokenizer file...")
               
                tokenizer_files = glob.glob('*.pickle')
                if tokenizer_files:
                    tokenizer_path = tokenizer_files[0]
                    with open(tokenizer_path, 'rb') as handle:
                        self.tokenizer = pickle.load(handle)
                    print(f" Tokenizer loaded from {tokenizer_path}")
                    return True
                else:
                    print(f" No tokenizer file found.")
                    return False
        except Exception as e:
            print(f" Error loading tokenizer: {e}")
            return False
    
    def predict_emotion(self, text):
       
        if self.model is None:
            print("Model not loaded. Cannot predict.")
            return None, None, None
        
        if self.tokenizer is None:
            print(" Tokenizer not loaded. Cannot preprocess text.")
            return None, None, None
        
        try:
            
            seq = self.tokenizer.texts_to_sequences([text])
            if not seq[0]:  # Empty sequence (all words are OOV)
                print(" Warning: No known words in text. Using <OOV> token.")
            
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
            
           
            probabilities = pred[0]
            
            return emotion, confidence, probabilities
        
        except Exception as e:
            print(f" Error during prediction: {e}")
            return None, None, None
    
    def predict_multiple(self, texts):
        
        results = []
        for text in texts:
            emotion, confidence, probs = self.predict_emotion(text)
            if emotion:
                results.append({
                    'text': text,
                    'emotion': emotion,
                    'confidence': confidence,
                    'probabilities': probs.tolist() if hasattr(probs, 'tolist') else probs
                })
        return results
    
    def predict_with_details(self, text):
      
        emotion, confidence, probabilities = self.predict_emotion(text)
        
        if emotion is None:
            return None
        
       
       
        print(f"Input Text: '{text}'")
        print(f"\nPredicted Emotion: {emotion.upper()}")
        print(f"Confidence: {confidence:.2%}")
        
        print(f"\nAll Emotion Probabilities:")
        for i, (label, prob) in enumerate(zip(self.labels, probabilities)):
            percentage = prob * 100
            bar = "" * int(percentage / 5)  # Each █ = 5%
            print(f"  {label:10s}: {percentage:6.2f}% {bar}")
        
        print(f"\nTop 3 Emotions:")
        top_indices = np.argsort(probabilities)[-3:][::-1]
        for idx in top_indices:
            print(f"  {self.labels[idx]}: {probabilities[idx]*100:.2f}%")
        
        return emotion, confidence, probabilities

if __name__ == "__main__":
    
    predictor = EmotionTextPredictor()
    
    if predictor.model and predictor.tokenizer:
      
        print(f"Available emotions: {predictor.labels}")
        
        
        
        sample_texts = [
            "I am so angry right now!",
            "I feel scared and anxious",
            "This makes me very happy",
            "I love this so much",
            "I am feeling very sad today",
            "What a surprising result!"
        ]
        
        for i, text in enumerate(sample_texts, 1):
            print(f"\n{i}. Text: '{text}'")
            emotion, confidence, _ = predictor.predict_emotion(text)
            if emotion:
                print(f"   → Emotion: {emotion} (Confidence: {confidence:.2%})")
        
       
        
        
        test_text = "I feel wonderful today!"
        emotion, confidence, _ = predictor.predict_emotion(test_text)
        print(f"Example: '{test_text}'")
        print(f"Prediction: {emotion} ({confidence:.2%})")
    else:
        print("\n Predictor initialization failed.")
       