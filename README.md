Modèle de Détection de Fautes entre les Images

Le modèle de détection de fautes entre les images est un programme Python conçu pour identifier et mettre en évidence les différences visuelles entre deux images. Il est particulièrement utile pour repérer les erreurs ou les altérations dans des paires d'images similaires.

Comment ça fonctionne :

Chargement des Images :

Le programme prend en entrée deux images que nous appellerons "image de référence" et "image à analyser".
Conversion en Niveaux de Gris :

Les images sont converties en images en niveaux de gris pour simplifier le traitement.
Calcul de la Différence Absolue :

En utilisant la bibliothèque OpenCV, le programme calcule la différence absolue entre les pixels des deux images. Cette différence capture les variations de luminosité et de couleur entre les images.
Application d'un Seuil :

Un seuil est appliqué à la différence calculée pour identifier les zones où la différence est significative. Cela permet de séparer les régions d'intérêt des régions similaires.
Détection des Contours :

Ensuite, le programme trouve les contours des zones où la différence est supérieure au seuil défini. Ces contours entourent les zones où les fautes ou altérations sont détectées.
Mise en Évidence des Fautes :

Enfin, les contours détectés sont superposés à l'image de référence. Cela crée une nouvelle image où les régions de fautes sont mises en évidence en rouge.
Utilisation :

Ce modèle peut être utilisé dans divers domaines tels que le contrôle qualité, la comparaison d'images médicales, la détection de mouvements, etc.
Il fournit une visualisation claire des différences entre les images, aidant les utilisateurs à identifier rapidement les fautes ou les altérations.
