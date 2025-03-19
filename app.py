from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import tensorflow as tf
from model.utils import (
    load_and_preprocess_image, improved_gradcam,
    create_heatmap_overlay, load_model_metrics,
    calculate_ensemble_prediction
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# Global variables for models
custom_cnn = None
resnet101 = None

def init_models():
    """Initialize models with proper error handling"""
    global custom_cnn, resnet101
    
    try:
        print("Loading Custom CNN model...")
        custom_cnn = load_model('models/custom_cnn_final.keras')
        # Print model summary to help identify correct layers
        print("Custom CNN layer names:")
        for i, layer in enumerate(custom_cnn.layers):
            print(f"{i}: {layer.name}")
        
        # Warm up the model with a dummy prediction
        dummy_input = np.zeros((1, 224, 224, 1))
        _ = custom_cnn.predict(dummy_input, verbose=0)
        print("Custom CNN model loaded successfully")
        
        print("Loading ResNet101 model...")
        resnet101 = load_model('models/resnet101_final.keras')
        # Print model summary to help identify correct layers
        print("ResNet101 layer names:")
        for i, layer in enumerate(resnet101.layers):
            print(f"{i}: {layer.name}")
        
        # Warm up the model with a dummy prediction
        dummy_input = np.zeros((1, 224, 224, 3))
        _ = resnet101.predict(dummy_input, verbose=0)
        print("ResNet101 model loaded successfully")
        
        # Print model summaries for debugging
        print("=== Custom CNN Model ===")
        print(f"Input shape: {custom_cnn.input_shape}")
        print(f"Output shape: {custom_cnn.output_shape}")
        
        print("=== ResNet101 Model ===")
        print(f"Input shape: {resnet101.input_shape}")
        print(f"Output shape: {resnet101.output_shape}")
        
        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Initialize models on startup
init_models()

@app.route('/')
def home():
    metrics = load_model_metrics()
    return render_template('index.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Check if models are loaded
        global custom_cnn, resnet101
        if custom_cnn is None or resnet101 is None:
            if not init_models():
                return jsonify({'error': 'Failed to initialize models'})
        
        # Save and process image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"Saved file to: {filepath}")

        # Read original image for overlay
        original_img = cv2.imread(filepath)
        if original_img is None:
            return jsonify({'error': 'Failed to read uploaded image'})
            
        original_img = cv2.resize(original_img, (224, 224))
        
        # Prepare images for both models
        img_gray = load_and_preprocess_image(filepath, is_rgb=False)
        img_rgb = load_and_preprocess_image(filepath, is_rgb=True)
        
        # Get predictions
        custom_pred = float(custom_cnn.predict(img_gray, verbose=0)[0][0])
        resnet_pred = float(resnet101.predict(img_rgb, verbose=0)[0][0])
        
        # Determine if pneumonia is detected
        custom_is_pneumonia = custom_pred > 0.5
        resnet_is_pneumonia = resnet_pred > 0.5
        
        # Calculate ensemble prediction
        ensemble_result = calculate_ensemble_prediction(custom_pred, resnet_pred)
        ensemble_is_pneumonia = ensemble_result['prediction'] == 'Pneumonia'
        
        # Original image URL - always shown
        original_image_url = url_for('static', filename=f'uploads/{filename}')
        
        # Initialize heatmap URLs with the original image URL
        custom_heatmap_url = original_image_url
        resnet_heatmap_url = original_image_url
        
        # Generate heatmaps if pneumonia is detected
        if custom_is_pneumonia:
            try:
                print("Generating custom CNN heatmap...")
                custom_heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], f'customheatmap_{filename}')
                custom_heatmap = improved_gradcam(custom_cnn, img_gray, is_custom_cnn=True)
                custom_overlay = create_heatmap_overlay(original_img, custom_heatmap)
                cv2.imwrite(custom_heatmap_path, custom_overlay)
                custom_heatmap_url = url_for('static', filename=f'uploads/customheatmap_{filename}')
                print(f"Custom heatmap saved to: {custom_heatmap_path}")
            except Exception as e:
                print(f"Error generating custom CNN heatmap: {str(e)}")
                import traceback
                traceback.print_exc()
                # If we still want a fallback heatmap for pneumonia cases
                from model.utils import create_medical_heatmap, create_heatmap_overlay
                fallback_heatmap = create_medical_heatmap(img_gray[0])
                custom_overlay = create_heatmap_overlay(original_img, fallback_heatmap)
                cv2.imwrite(custom_heatmap_path, custom_overlay)
                custom_heatmap_url = url_for('static', filename=f'uploads/customheatmap_{filename}')
        
        if resnet_is_pneumonia:
            try:
                print("Generating ResNet heatmap...")
                resnet_heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], f'resnetheatmap_{filename}')
                resnet_heatmap = improved_gradcam(resnet101, img_rgb, is_custom_cnn=False)
                resnet_overlay = create_heatmap_overlay(original_img, resnet_heatmap)
                cv2.imwrite(resnet_heatmap_path, resnet_overlay)
                resnet_heatmap_url = url_for('static', filename=f'uploads/resnetheatmap_{filename}')
                print(f"ResNet heatmap saved to: {resnet_heatmap_path}")
            except Exception as e:
                print(f"Error generating ResNet heatmap: {str(e)}")
                import traceback
                traceback.print_exc()
                # If we still want a fallback heatmap for pneumonia cases
                from model.utils import create_medical_heatmap, create_heatmap_overlay
                fallback_heatmap = create_medical_heatmap(img_rgb[0], is_rgb=True)
                resnet_overlay = create_heatmap_overlay(original_img, fallback_heatmap)
                cv2.imwrite(resnet_heatmap_path, resnet_overlay)
                resnet_heatmap_url = url_for('static', filename=f'uploads/resnetheatmap_{filename}')
        
        # Return the results
        return jsonify({
            'custom_cnn': {
                'probability': custom_pred,
                'prediction': 'Pneumonia' if custom_is_pneumonia else 'Normal',
                'heatmap': custom_heatmap_url,  # Now always has a value (original image or heatmap)
                'show_heatmap': True  # Always show the heatmap area
            },
            'resnet101': {
                'probability': resnet_pred,
                'prediction': 'Pneumonia' if resnet_is_pneumonia else 'Normal',
                'heatmap': resnet_heatmap_url,  # Now always has a value (original image or heatmap)
                'show_heatmap': True  # Always show the heatmap area
            },
            'ensemble': ensemble_result,
            'show_ensemble_heatmap': True,  # Always show the ensemble section
            'original_image': original_image_url
        })

    except Exception as e:
        import traceback
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)