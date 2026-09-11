# SSTF

Official implementation of the SSTF framework for cattle behavior classification.

## Environment

```bash
pip install -r requirements.txt
```

### requirements.txt

```
torch>=1.10.0
torchvision>=0.11.0
git+https://github.com/openai/CLIP.git
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
thop>=0.0.31
```

## Repository Structure

```
├── main.py                     # Entry point for training and evaluation
├── config.py                   # All hyperparameters and arguments
├── model.py                    # SSTF model and ablation variants
├── fbm_paper_components.py     # TFM components (Trend, Seasonal, Interaction)
├── clip_integration.py         # Frozen CLIP encoder and text projection
├── gpt_text_generator.py       # GPT-OSS offline text generation
├── data.py                     # Data loading, windowing, train/val/test split
├── train.py                    # Training and validation loops
├── test.py                     # Evaluation and t-SNE visualization
├── draw.py                     # Confusion matrices, loss curves, weight distributions
├── early_stopping.py           # Early stopping with checkpoint saving
├── set_random_seed.py          # Seed setting for reproducibility
├── text_utils.py               # CSV save/load for generated text descriptions
└── requirements.txt            # Python dependencies
```

## Usage
### Train the full SSTF model
```
python main.py --model HybridFBM_LSTM_CNN_2D_Text_Dynamic_Contrastive
```

