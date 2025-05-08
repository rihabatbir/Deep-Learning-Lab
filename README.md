# Projet : DL models optimization : Pruning, Quantization, Knowledge Distillation 

Ce projet a pour but d'explorer trois techniques populaires permettant d'optimiser les modèles de deep learning afin de les rendre plus rapides et moins gourmands en ressources, sans sacrifier (ou très peu) la précision.

Les trois méthodes étudiées sont :

- Distillation de connaissances
- Pruning (élagage)
- Quantification (quantization)

---

# Fichiers inclus

`KD.ipynb`: Mise en œuvre d’une distillation de connaissances                   
`pruning.ipynb`: Réduction de la complexité d’un réseau via suppression de poids     
`quantization.ipynb`: Compression du modèle par réduction de la précision des données    

---

# Détails des notebooks

# 1. `KD.ipynb` – Distillation de Connaissances

Ce notebook montre comment un modèle simple (appelé étudiant) peut être entraîné à imiter un modèle plus complexe (appelé enseignant). Le but est de transférer les compétences du modèle lourd vers un plus léger.

**Étapes principales :**
- Chargement du dataset MNIST
- Entraînement d’un modèle enseignant avec haute capacité
- Création d’un modèle plus petit
- Entraînement de l’étudiant en combinant deux pertes :
  - Une perte classique par rapport aux étiquettes réelles
  - Une perte entre les prédictions de l'étudiant et celles de l’enseignant (avec température)
- Évaluation comparative des performances

---

# 2. `pruning.ipynb` – Pruning

L’objectif ici est de supprimer les connexions inutiles dans un réseau, pour le rendre plus léger.

**Étapes principales :**
- Préparation du dataset MNIST
- Entraînement d’un modèle classique
- Application de l’élagage sur certaines couches (pruning global ou ciblé)
- Réentraînement pour compenser la perte potentielle de performance
- Mesures : taux de sparsité, précision avant/après, impact sur la taille

---

# 3. `quantization.ipynb` – Quantization

Ce notebook aborde une technique de compression qui consiste à représenter les poids et activations avec une précision plus faible (ex. int8 au lieu de float32).

**Étapes principales :**
- Entraînement d’un modèle sur MNIST
- Application d’une quantification dynamique avec PyTorch
- Évaluation du modèle quantifié :
  - Comparaison des performances
  - Réduction de la taille du fichier modèle
  - Gain en vitesse d’exécution

---

# Présentation du dataset : MNIST

MNIST est un jeu de données bien connu en reconnaissance d’images. Il contient 70 000 images en noir et blanc de chiffres manuscrits (28x28 pixels). Chaque image correspond à un chiffre entre 0 et 9.

- **60 000 exemples pour l’entraînement**
- **10 000 pour les tests**

Idéal pour des tâches de classification simples et pour tester des techniques de compression.

---

# Environnement utilisé

- **Langage :** Python 3.10+
- **Plateforme :** Jupyter Notebook
- **Bibliothèques principales :**
  - PyTorch (`torch`, `torchvision`)
  - `matplotlib`
  - `numpy`

# Installer les dépendances :

```bash
pip install torch torchvision matplotlib numpy
