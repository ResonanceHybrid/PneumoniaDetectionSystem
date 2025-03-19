# Save this as layer_finder.py and run it once to find the right layers for GradCAM
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def init_models():
    """Initialize models with proper error handling and architecture debugging"""
    global custom_cnn, resnet101
    
    try:
        print("Loading Custom CNN model...")
        custom_cnn = load_model('models/custom_cnn_final.keras')
        
        # Debug: Print Custom CNN architecture
        print("=== Custom CNN Architecture ===")
        for i, layer in enumerate(custom_cnn.layers):
            print(f"{i}: {layer.name}, type: {type(layer).__name__}, " + 
                  f"output shape: {layer.output_shape}")
        
        # Find suitable GradCAM layers for Custom CNN
        suitable_layers = []
        for i, layer in enumerate(custom_cnn.layers):
            if 'conv2d' in layer.name.lower() and i < len(custom_cnn.layers) - 3:
                suitable_layers.append((i, layer.name))
        print(f"Suitable GradCAM layers for Custom CNN: {suitable_layers}")
        
        # Warm up the model with a dummy prediction
        dummy_input = np.zeros((1, 224, 224, 1))
        _ = custom_cnn.predict(dummy_input, verbose=0)
        print("Custom CNN model loaded successfully")
        
        print("Loading ResNet101 model...")
        resnet101 = load_model('models/resnet101_final.keras')
        
        # Debug: Print ResNet architecture
        print("=== ResNet101 Architecture ===")
        for i, layer in enumerate(resnet101.layers):
            print(f"{i}: {layer.name}, type: {type(layer).__name__}, " + 
                  f"output shape: {layer.output_shape}")
        
        # Find suitable GradCAM layers for ResNet
        suitable_layers = []
        for i, layer in enumerate(resnet101.layers):
            if ('conv' in layer.name.lower() or 'block' in layer.name.lower()) and i < len(resnet101.layers) - 3:
                suitable_layers.append((i, layer.name))
        print(f"Suitable GradCAM layers for ResNet: {suitable_layers[-5:]}")  # Last 5 suitable layers
        
        # Warm up the model with a dummy prediction
        dummy_input = np.zeros((1, 224, 224, 3))
        _ = resnet101.predict(dummy_input, verbose=0)
        print("ResNet101 model loaded successfully")
        
        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False