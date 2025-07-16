# ntb-redux

LM powered bot to play the video game nuclear throne

## Dependencies

### Ubuntu 24 LTS

#### For Screenshots

- `sudo apt install scrot wmctrl`

#### For Keyboard and Mouse Capture

- `sudo apt install xinput xev`
  Then make sure user has access to input devices

### Cleaning and Training

```
# Run the cleaning script
npx run clean-training-data.ts ./training_data ./cleaned_data --val-split 0.2 --test-split 0.1

# This will:
# - Process all sessions in ./training_data
# - Filter out low-quality frames
# - Split data into train/validation/test
# - Create PyTorch-ready datasets
# - Copy all screenshots to a centralized location
```

# Install TensorFlow.js with GPU support

`npm install -s @tensorflow/tfjs-node-gpu @types/node`

# OR for CPU only:

`npm install @tensorflow/tfjs-node @types/node`

# Start with a smaller model for quick testing

npx tsx ./src/tfjs\_-_training_setup.ts ./cleaned_data --model custom_cnn --epochs 10 --batch-size 4

# Or for a more thorough training run

npx tsx ./src/tfjs-training-setup.ts ./cleaned_data --model custom_cnn --epochs 50 --batch-size 8 --learning-rate 0.001
