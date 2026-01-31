# Dissertation_Multimodal_transformer

* This project aims to build a **multimodal emotion recognition system** using the MELD and RAVDESS dataset by combining information from **text, audio, and visual modalities**.

* The workflow includes **data preprocessing**, modality-specific **feature extraction**, and deep learning-based **multi-class emotion classification**.

* A major focus of the work is handling **class imbalance**, which significantly affects minority emotion recognition in real-world datasets.

* Techniques such as **Weighted Random Sampling** are incorporated to improve balanced performance, particularly reflected through Macro-F1 evaluation.

* The repository is structured into clear modules for **training, inference, feature extraction, and deployment**, ensuring reproducibility and extensibility.



## Repository Structure

```bash
project_root/
|
|-- README.md
|   `-- Main documentation describing the project objective, workflow, and usage.
|
|-- extracted_embeddings/
|   `-- Stores precomputed multimodal embeddings for faster training and reuse.
|   |
|   `-- Features/
|       |-- audio.zip
|       |   `-- Compressed audio embeddings extracted from MELD utterances.
|       |
|       |-- text.zip
|       |   `-- Compressed text embeddings extracted from dialogue transcripts.
|       |
|       `-- vision.zip
|           `-- Compressed visual embeddings extracted from video frames/faces.
|
|-- model/
|   `-- Contains trained model checkpoints for emotion classification.
|   |
|   |-- multimodal_emotion_model.pth
|   |   `-- Baseline multimodal model trained with standard sampling.
|   |
|   `-- Wrs_multimodal_emotion_model.pth
|       `-- Model trained with Weighted Random Sampling to handle imbalance.
|
`-- src/
    |
    |-- Data_preprocessing/
    |   `-- Clean_preprocess_MELD_Data.ipynb
    |       `-- Cleans and prepares the MELD dataset for feature extraction.
    |
    |-- feature_extraction/
    |   |-- extract_audio_features.ipynb
    |   |   `-- Extracts acoustic representations from speech signals.
    |   |
    |   |-- extract_text_features.ipynb
    |   |   `-- Extracts semantic embeddings using NLP models.
    |   |
    |   |-- extract_visual_features.ipynb
    |   |   `-- Extracts visual embeddings from video frames/faces.
    |   |
    |   |-- extract_labels_and_analysis.ipynb
    |   |   `-- Performs class imbalance distribution analysis.
    |   |
    |   |-- only_audio_extract_features.ipynb
    |   |   `-- Audio-only feature extraction experiments.
    |   |
    |   `-- only_extract_text_features.ipynb
    |       `-- Text-only feature extraction experiments.
    |
    |-- training/
    |   |-- train.ipynb
    |   |   `-- Baseline training using stratified sampling.
    |   |
    |   `-- WtRandomSampler_train.ipynb
    |       `-- Training with Weighted Random Sampling for imbalance handling.
    |
    `-- inference/
        `-- inference.ipynb
            `-- Runs evaluation and emotion prediction on unseen samples.
