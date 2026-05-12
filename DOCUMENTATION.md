# 🚀 Scientific RAG Local : Documentation Technique

Bienvenue dans la documentation de votre assistant de recherche scientifique local. Ce document explique comment fonctionne la machine sous le capot et pourquoi elle est si performante sur votre configuration (RTX 4070 Ti / 64 Go RAM).

---

## 🏗️ La "Stack" : Les composants du système

Pour construire cette application, nous avons assemblé plusieurs briques technologiques de pointe :

### 1. L'Orchestrateur : **LangChain**
*   **Rôle :** Le chef d'orchestre.
*   **Pourquoi lui ?** LangChain lie tous les composants (Ollama, ChromaDB, Docling). Il gère la "mémoire" de la conversation et s'assure que vos questions sont correctement reformulées avant d'interroger la base de données.

### 2. Le Cerveau d'Extraction : **Docling** (IBM Research)
*   **Rôle :** Lire vos fichiers PDF, Word ou PowerPoint.
*   **Pourquoi lui ?** Contrairement aux lecteurs classiques, Docling comprend la structure scientifique (tableaux, hiérarchie). Il transforme un document visuel en un format "Markdown" structuré.

### 3. La Mémoire à Long Terme : **ChromaDB**
*   **Rôle :** Stocker l'index vectoriel de vos documents.
*   **Fonctionnement :** Chaque paragraphe est transformé en un vecteur mathématique. Quand vous posez une question, ChromaDB cherche les paragraphes les plus proches de votre demande.

### 4. L'Intelligence d'Analyse : **Mistral-Nemo (12B)**
*   **Rôle :** Comprendre vos questions et rédiger des réponses rigoureuses.
*   **Avantage :** Optimisé par NVIDIA et Mistral pour le raisonnement technique. Tout tourne localement sur votre carte graphique (RTX 4070 Ti).

### 5. L'Interface Utilisateur : **React & Vite**
*   **Rôle :** Expérience fluide "Compositor v2.4".
*   **Bonus :** Support natif de KaTeX (formules) et visualiseur hybride (PDF original ou rendu natif Word).

---

## 🛠️ Nos Solutions Techniques "Spéciales"

### 💉 L'Ingestion Chirurgicale (Page-par-Page)
Pour les PDFs, nous avons créé un système qui découpe le fichier virtuellement pour le traiter **page par page**. Cela évite les erreurs de mémoire (`bad_alloc`) et garantit que le numéro de page cité est exact.

### 🖼️ Visualisation Hybride Haute-Fidélité
L'interface détecte le format du document :
*   **PDF :** Affiche le document original.
*   **Word (DOCX) :** Utilise `docx-preview` pour reconstruire visuellement le document dans votre navigateur, 100% localement.

### 🧠 Mémoire Contextuelle (Query Rewriting)
Le système ne se contente pas de chercher vos mots-clés. Il utilise une phase de "reformulation" pour comprendre le lien entre vos questions successives (ex: comprendre que "un exemple ?" se réfère au sujet précédent).

---

## 🚦 Comment l'utiliser au quotidien ?

1.  **Lancer :** Double-cliquez sur `start.bat`.
2.  **Importer :** Utilisez le bouton "Importer" pour ajouter vos fichiers (PDF, DOCX, PPTX).
3.  **Interroger :** Posez des questions techniques.
4.  **Vérifier :** Cliquez sur les badges de source (ex: `p.37` ou `DOC`) pour voir la preuve directement sur la droite.
5.  **Gérer :** Utilisez la barre latérale gauche pour retrouver ou supprimer vos anciennes discussions.

---
*[WFGY] Zone: SAFE | Documentation mise à jour pour la version v2.4 Stable*
