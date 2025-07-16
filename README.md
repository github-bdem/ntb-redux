# ntb-redux

LM powered bot to play the video game nuclear throne

## Dependencies

### Ubuntu 24 LTS

#### For Screenshots

- `sudo apt install scrot wmctrl xdotool`

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

npm run train ./cleaned_data --model custom_cnn --epochs 10

🛡️ Safety Features

Emergency Stop: Ctrl+C stops everything immediately
Safety Mode: Automatically stops on errors
No-Controller Mode: Test predictions without actual input
Confidence Threshold: Only acts when confident
Dead Zones: Prevents jittery movements

📊 Performance Tuning

FPS: Start with --fps 10, increase as needed
Confidence: Use --confidence 0.3 for more actions, 0.6 for fewer
Smoothing: --smoothing 0.1 for responsive, 0.5 for smooth
Mouse Speed: Adjust --mouse-speed based on your game sensitivity

🐛 Troubleshooting
If the AI can't find the game window:

Make sure Nuclear Throne is running
Try different window titles: --window nuclear or --window throne

If inputs don't work:

Check that xdotool is installed: sudo apt install xdotool
Test with --no-controller first
Make sure the game window is focused

If performance is poor:

Lower FPS: --fps 10
Reduce batch size during training
Use --no-controller for testing
