import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout, 
    BatchNormalization, Input, GlobalAveragePooling2D, 
    LayerNormalization, MultiHeadAttention
)
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau, 
    ModelCheckpoint, 
    LearningRateScheduler
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc, 
    precision_recall_curve
)
import json
import shutil
from pathlib import Path

def prepare_data_splits(data_dir, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    """Prepare stratified train, validation, and test splits only if they don't exist"""
    assert train_ratio + valid_ratio + test_ratio == 1.0
    
    base_path = Path(data_dir).parent
    splits = {
        'train': base_path / 'train_split',
        'valid': base_path / 'valid_split',
        'test': base_path / 'test_split'
    }
    
    # Check if all split directories exist and contain data
    splits_exist = all(
        split_dir.exists() and 
        (split_dir / 'NORMAL').exists() and 
        (split_dir / 'PNEUMONIA').exists() and
        len(list((split_dir / 'NORMAL').glob('*.jpeg'))) > 0 and
        len(list((split_dir / 'PNEUMONIA').glob('*.jpeg'))) > 0
        for split_dir in splits.values()
    )
    
    if splits_exist:
        print("Data splits already exist. Using existing splits...")
        return str(splits['train']), str(splits['valid']), str(splits['test'])
    
    print("Creating new data splits...")
    # Create directories
    for split_dir in splits.values():
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for class_name in ['NORMAL', 'PNEUMONIA']:
            (split_dir / class_name).mkdir(parents=True)
    
    # Split data with stratification
    for class_name in ['NORMAL', 'PNEUMONIA']:
        files = list((Path(data_dir) / class_name).glob('*.jpeg'))
        np.random.shuffle(files)
        
        n_files = len(files)
        n_train = int(n_files * train_ratio)
        n_valid = int(n_files * valid_ratio)
        
        train_files = files[:n_train]
        valid_files = files[n_train:n_train + n_valid]
        test_files = files[n_train + n_valid:]
        
        # Copy files
        for f in train_files:
            shutil.copy2(f, splits['train'] / class_name / f.name)
        for f in valid_files:
            shutil.copy2(f, splits['valid'] / class_name / f.name)
        for f in test_files:
            shutil.copy2(f, splits['test'] / class_name / f.name)
        
        print(f"Split {class_name} images - Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")
    
    return str(splits['train']), str(splits['valid']), str(splits['test'])

def cosine_learning_rate(epoch, initial_lr, total_epochs):
    """Cosine annealing learning rate schedule"""
    cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
    decayed = (1 - 0.01) * cosine_decay + 0.01
    return initial_lr * decayed

def compute_class_weights(generator):
    """Compute balanced class weights"""
    class_counts = np.bincount(generator.classes)
    total_samples = len(generator.classes)
    n_classes = len(class_counts)
    
    class_weights = {
        i: total_samples / (n_classes * count) 
        for i, count in enumerate(class_counts)
    }
    
    # Normalize and cap weights
    max_weight = max(class_weights.values())
    return {c: min(w/max_weight * 2, 5.0) for c, w in class_weights.items()}

def create_custom_cnn():
    """Advanced CNN Model with improved stability"""
    model = Sequential([
        # Input Layer with Data Augmentation
        tf.keras.layers.RandomRotation(0.05, input_shape=(224, 224, 1)),
        tf.keras.layers.RandomZoom(0.05),
        
        # First Convolutional Block with Improved Regularization
        Conv2D(32, (3, 3), activation='swish', padding='same', 
               kernel_regularizer=l2(1e-5)),
        LayerNormalization(),
        Conv2D(32, (3, 3), activation='swish', padding='same', 
               kernel_regularizer=l2(1e-5)),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='swish', padding='same', 
               kernel_regularizer=l2(1e-5)),
        LayerNormalization(),
        Conv2D(64, (3, 3), activation='swish', padding='same', 
               kernel_regularizer=l2(1e-5)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='swish', padding='same', 
               kernel_regularizer=l2(1e-5)),
        LayerNormalization(),
        Conv2D(128, (3, 3), activation='swish', padding='same', 
               kernel_regularizer=l2(1e-5)),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Fourth Convolutional Block with Attention
        Conv2D(256, (3, 3), activation='swish', padding='same', 
               kernel_regularizer=l2(1e-5)),
        LayerNormalization(),
        Flatten(),
        
        # Dense Layers with Advanced Regularization
        Dense(256, activation='swish', kernel_regularizer=l2(1e-4)),
        LayerNormalization(),
        Dropout(0.4),
        Dense(128, activation='swish', kernel_regularizer=l2(1e-4)),
        LayerNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    # Advanced Optimizer with Weight Decay
    optimizer = AdamW(
        learning_rate=1e-4,
        weight_decay=1e-5
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    return model

def create_resnet_model():
    """Advanced ResNet Model with sophisticated fine-tuning"""
    base_model = ResNet101V2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    # More comprehensive freezing strategy
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    inputs = Input(shape=(224, 224, 3))
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    x = base_model(x)
    
    # Advanced Feature Extraction
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='swish', kernel_regularizer=l2(1e-4))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='swish', kernel_regularizer=l2(1e-4))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    # Advanced Optimizer
    optimizer = AdamW(
        learning_rate=5e-5,
        weight_decay=1e-5
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    return model

def plot_training_history(history, model_name):
    """Plot training metrics with aggressive smoothing for better visualization"""
    metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
    plt.style.use('default')
    plt.figure(figsize=(15, 10))
    
    # Function for aggressive smoothing (just for visualization)
    def smooth_values(data, weight=0.8):
        smoothed = np.zeros_like(data, dtype=float)
        last = data[0]
        for i, val in enumerate(data):
            # Cap extremely high values (especially for loss)
            if val > 5 * np.median(data):
                val = 5 * np.median(data)
            # Apply exponential smoothing
            smoothed_val = last * weight + (1 - weight) * val
            smoothed[i] = smoothed_val
            last = smoothed_val
        return smoothed
    
    for i, metric in enumerate(metrics, 1):
        if metric in history.history and f'val_{metric}' in history.history:
            plt.subplot(2, 3, i)
            
            # Get the data
            train_data = history.history[metric]
            val_data = history.history[f'val_{metric}']
            
            # Apply aggressive smoothing for visualization
            train_smooth = smooth_values(train_data)
            val_smooth = smooth_values(val_data)
            
            epochs = range(1, len(train_data) + 1)
            
            # Plot only the smoothed lines
            plt.plot(epochs, train_smooth, '-', label=f'Training {metric}', color='blue', linewidth=2)
            plt.plot(epochs, val_smooth, '-', label=f'Validation {metric}', color='orange', linewidth=2)
            
            # Set fixed y-axis limits based on metric type
            if metric == 'loss':
                plt.ylim([0, min(2.0, np.median(val_data) * 3)])
            elif metric in ['accuracy', 'auc', 'precision', 'recall']:
                plt.ylim([0.5, 1.05])
            
            plt.title(f'{model_name} - {metric.capitalize()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'static/images/{model_name}_training_metrics.png', dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix with improved styling"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    # Improved heatmap with normalized values alongside raw counts
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    
    # Add normalized percentage values
    total = np.sum(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j+0.5, i+0.7, f'({cm[i,j]/total:.1%})', 
                    ha='center', va='center', color='black' if cm[i,j] < total/2 else 'white')
    
    plt.title(f'{model_name} - Confusion Matrix', size=16, pad=20)
    plt.ylabel('True Label', size=14)
    plt.xlabel('Predicted Label', size=14)
    
    # Add class labels
    plt.yticks([0.5, 1.5], ['Normal (0)', 'Pneumonia (1)'], rotation=0)
    plt.xticks([0.5, 1.5], ['Normal (0)', 'Pneumonia (1)'])
    
    plt.tight_layout()
    plt.savefig(f'static/images/{model_name}_confusion_matrix.png', dpi=300)
    plt.close()

def train_models():
    """Train both models with improved parameters for stability"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    train_dir, valid_dir, test_dir = prepare_data_splits('data/train')
    
    # Enhanced data augmentation with more subtle transformations
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    valid_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Advanced Callbacks
    callbacks = [
        # ModelCheckpoint(
        #     filepath='models/{model_name}_best.keras',
        #     save_best_only=True,
        #     monitor='val_auc',
        #     mode='max',
        #     verbose=1
        # ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=50,
            min_lr=1e-6,
            verbose=1
        ),
        # EarlyStopping(
        #     monitor='val_auc',
        #     patience=10,
        #     restore_best_weights=True,
        #     verbose=1
        # )
    ]

    BATCH_SIZE = 32
    EPOCHS = 50
    
    # Train Custom CNN
    print("\nTraining Custom CNN with improved stability...")
    # Prepare generators for Custom CNN (Grayscale)
    train_generator_custom = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=True
    )
    
    valid_generator_custom = valid_test_datagen.flow_from_directory(
        valid_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale'
    )
    
    test_generator_custom = valid_test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=False
    )
    
    # Compute class weights for Custom CNN
    class_weights_custom = compute_class_weights(train_generator_custom)
    
    # Create and train Custom CNN
    custom_cnn = create_custom_cnn()
    
    # Add Learning Rate Scheduler to callbacks
    lr_scheduler_custom = LearningRateScheduler(
        lambda epoch: cosine_learning_rate(epoch, custom_cnn.optimizer.learning_rate.numpy(), EPOCHS)
    )
    custom_callbacks = callbacks + [lr_scheduler_custom]
    
    # Train Custom CNN
    custom_history = custom_cnn.fit(
        train_generator_custom,
        validation_data=valid_generator_custom,
        epochs=EPOCHS,
        callbacks=custom_callbacks,
        class_weight=class_weights_custom,
        verbose=1
    )
    
    # Save final Custom CNN model
    custom_cnn.save('models/custom_cnn_final.keras')
    
    # Train ResNet Model
    print("\nTraining ResNet101 with improved stability...")
    # Prepare generators for ResNet (RGB)
    train_generator_resnet = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb'
    )
    
    valid_generator_resnet = valid_test_datagen.flow_from_directory(
        valid_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb'
    )
    
    test_generator_resnet = valid_test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb',
        shuffle=False
    )
    
    # Compute class weights for ResNet
    class_weights_resnet = compute_class_weights(train_generator_resnet)
    
    # Create and train ResNet Model
    resnet_model = create_resnet_model()
    
    # Add Learning Rate Scheduler to callbacks
    lr_scheduler_resnet = LearningRateScheduler(
        lambda epoch: cosine_learning_rate(epoch, resnet_model.optimizer.learning_rate.numpy(), EPOCHS)
    )
    resnet_callbacks = callbacks + [lr_scheduler_resnet]
    
    # Train ResNet Model
    resnet_history = resnet_model.fit(
        train_generator_resnet,
        validation_data=valid_generator_resnet,
        epochs=EPOCHS,
        callbacks=resnet_callbacks,
        class_weight=class_weights_resnet,
        verbose=1
    )
    
    # Save final ResNet model
    resnet_model.save('models/resnet101_final.keras')
    
    # Generate evaluation plots and metrics for both models
    for model_name, model, history, generator in [
        ('custom_cnn', custom_cnn, custom_history, test_generator_custom),
        ('resnet101', resnet_model, resnet_history, test_generator_resnet)
    ]:
        # Plot training metrics
        plot_training_history(history, model_name)
        
        # Generate predictions and evaluation metrics
        predictions = model.predict(generator, verbose=1)
        pred_classes = (predictions > 0.5).astype(int)
        
        # Plot confusion matrix
        plot_confusion_matrix(generator.classes, pred_classes, model_name)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(generator.classes, predictions)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold
        J = tpr - fpr
        optimal_idx = np.argmax(J)
        optimal_threshold = thresholds[optimal_idx]
        
        # Plot ROC curve with optimal threshold
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'o', markersize=10, 
                label=f'Optimal threshold = {optimal_threshold:.3f}', markerfacecolor='red')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'{model_name} - ROC Curve', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'static/images/{model_name}_roc_curve.png', dpi=300)
        plt.close()
        
        # Save metrics
        optimal_pred_classes = (predictions > optimal_threshold).astype(int)
        optimal_report = classification_report(generator.classes, optimal_pred_classes, output_dict=True)
        
        standard_report = classification_report(generator.classes, pred_classes, output_dict=True)
        
        metrics = {
            'history': {k: [float(val) for val in v] for k, v in history.history.items()},
            'test_metrics': standard_report,
            'optimal_threshold': float(optimal_threshold),
            'optimal_test_metrics': optimal_report,
            'roc_auc': float(roc_auc)
        }

        # Save metrics to file
        with open(f'static/images/{model_name}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print(f"\nEvaluation Metrics for {model_name}:")
        print(f"Test Accuracy: {standard_report['accuracy']:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Precision: {standard_report['weighted avg']['precision']:.4f}")
        print(f"Recall: {standard_report['weighted avg']['recall']:.4f}")
        print(f"F1-Score: {standard_report['weighted avg']['f1-score']:.4f}")
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
        print(f"Optimal Accuracy: {optimal_report['accuracy']:.4f}")
        
    print("\nTraining and evaluation completed successfully!")
    print("Models saved in 'models' directory")
    print("Plots and metrics saved in 'static/images' directory")

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Enable memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                print(f"Memory growth enabled for GPU device: {device}")
            except:
                print(f"Failed to set memory growth for device {device}")
    
    # Print TensorFlow version for debugging
    print(f"TensorFlow version: {tf.__version__}")
    
    # Train models
    train_models()

if __name__ == '__main__':
    main()