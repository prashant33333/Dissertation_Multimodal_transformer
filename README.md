# Dissertation_Multimodal_transformer

* This project aims to build a **multimodal emotion recognition system** using the MELD and RAVDESS dataset by combining information from **text, audio, and visual modalities**.

* The workflow includes **data preprocessing**, modality-specific **feature extraction**, and deep learning-based **multi-class emotion classification**.

* A major focus of the work is handling **class imbalance**, which significantly affects minority emotion recognition in real-world datasets.

* Techniques such as **Weighted Random Sampling** are incorporated to improve balanced performance, particularly reflected through Macro-F1 evaluation.

* The repository is structured into clear modules for **training, inference, feature extraction, and deployment**, ensuring reproducibility and extensibility.



#Codebase structure


project_root/
│
├── README.md
│   └── Main documentation describing the project objective, workflow, and usage.
│
├── extracted_embeddings/
│   └── Stores precomputed multimodal feature embeddings for faster training and reuse.
│
│   └── Features/
│       ├── audio.zip
│       │   └── Compressed audio modality embeddings extracted from MELD utterances.
│       │
│       ├── text.zip
│       │   └── Compressed textual embeddings extracted from dialogue transcripts.
│       │
│       └── vision.zip
│           └── Compressed visual feature embeddings extracted from video frames/faces.
│
├── model/
│   └── Contains trained model checkpoints for emotion classification.
│
│   ├── multimodal_emotion_model.pth
│   │   └── Baseline multimodal emotion recognition model trained with standard sampling.
│   │
│   └── Wrs_multimodal_emotion_model.pth
│       └── Improved model trained using Weighted Random Sampling to handle imbalance.
│
└── src/
    └── Source notebooks organized by pipeline stage.
    
    ├── Data_preprocessing/
    │   └── Clean_preprocess_MELD_Data.ipynb
    │       └── Cleans, formats, and prepares MELD dataset for feature extraction.
    │
    ├── feature_extraction/
    │   ├── extract_audio_features.ipynb
    │   │   └── Extracts acoustic representations from speech signals.
    │   │
    │   ├── extract_text_features.ipynb
    │   │   └── Extracts semantic embeddings from utterance transcripts using NLP models.
    │   │
    │   ├── extract_visual_features.ipynb
    │   │   └── Extracts facial/frame-level visual embeddings from video modality.
    │   │
    │   ├── extract_labels_and_analysis.ipynb
    │   │   └── Loads emotion labels and performs class imbalance distribution analysis.
    │   │
    │   ├── only_audio_extract_features.ipynb
    │   │   └── Standalone pipeline for audio-only feature extraction experiments.
    │   │
    │   └── only_extract_text_features.ipynb
    │       └── Standalone pipeline for text-only feature extraction experiments.
    │
    ├── training/
    │   ├── train.ipynb
    │   │   └── Baseline multimodal training notebook using stratified batch sampling.
    │   │
    │   └── WtRandomSampler train.ipynb
    │       └── Training notebook implementing Weighted Random Sampling for imbalance-aware learning.
    │
    └── inference/
        └── inference.ipynb
            └── Runs evaluation and generates predictions on unseen multimodal emotion samples.

