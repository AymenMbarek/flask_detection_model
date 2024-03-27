import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model(input_shape):
    # Créer un modèle séquentiel
    model = models.Sequential()

    # Ajouter des couches convolutionnelles
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Ajouter des couches denses à la fin
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # 1 pour la classification binaire

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def prepare_data(directory, target_size, batch_size):
    # Création des générateurs de données
    datagen = ImageDataGenerator(rescale=1./255)
    
    # Préparation des générateurs de flux de données d'images
    train_generator = datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary')  # binary pour la comparaison 1 vs 1

    return train_generator

# Paramètres du modèle
input_shape = (250, 250, 3)  # Exemple de forme d'entrée
target_size = (250, 250)     # Doit correspondre à input_shape
batch_size = 32              # Taille du lot pour l'entraînement

# Construire le modèle
model = build_model(input_shape)

# Préparer les données
train_directory = 'C:/Users/Darkk/Desktop/wejden_mars/model/data'
train_generator = prepare_data(train_directory, target_size, batch_size)

# Entraîner le modèle
model.fit(train_generator, epochs=10)  # Exécute 10 époques d'entraînement
