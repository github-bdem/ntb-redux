#!/usr/bin/env ts-node

import * as tf from '@tensorflow/tfjs-node-gpu'; // For GPU support
// Alternative: import * as tf from '@tensorflow/tfjs-node'; // CPU only
import { promises as fs } from 'fs';
import { join } from 'path';
import * as path from 'path';
import { fileURLToPath } from 'url';

interface TrainingConfig {
  dataDir: string;
  modelType: 'efficientnet' | 'mobilenet' | 'custom_cnn';
  epochs: number;
  batchSize: number;
  learningRate: number;
  validationSplit: number;
  savePath: string;
  resumeFrom?: string;
}

interface DatasetSample {
  screenshot: string;
  outputs: {
    movement_x: number;
    movement_y: number;
    aim_x: number;
    aim_y: number;
    shooting: number;
  };
  timestamp: number;
}

class TensorFlowTrainer {
  private config: TrainingConfig;
  private model: tf.LayersModel | null = null;
  private trainDataset: tf.data.Dataset<{ xs: tf.Tensor; ys: tf.Tensor }> | null = null;
  private valDataset: tf.data.Dataset<{ xs: tf.Tensor; ys: tf.Tensor }> | null = null;

  constructor(config: TrainingConfig) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    console.log('🔧 Initializing TensorFlow.js training...');

    // Configure for optimal performance
    if (tf.getBackend() === 'tensorflow') {
      // For ROCm/GPU backend
      tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
      tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
      tf.env().set('WEBGL_PACK', true);

      // Set memory growth for GPU
      try {
        const gpuMemoryInfo = tf.memory();
        console.log(`💾 Initial GPU memory: ${JSON.stringify(gpuMemoryInfo)}`);
      } catch (e) {
        console.log('⚠️  GPU memory info not available');
      }
    }

    console.log(`🖥️  Backend: ${tf.getBackend()}`);
    console.log(`📊 TensorFlow.js version: ${tf.version.tfjs}`);

    // Verify we can create tensors
    try {
      const testTensor = tf.zeros([1, 3, 240, 320]);
      console.log(`✅ Test tensor created: ${testTensor.shape}`);
      testTensor.dispose();
    } catch (error) {
      throw new Error(`Failed to create test tensor: ${error}`);
    }

    // Load datasets
    await this.loadDatasets();

    // Create model
    this.model = await this.createModel();

    console.log('✅ Initialization complete');
    console.log(`📈 Model parameters: ${this.model.countParams()}`);
  }

  private async loadDatasets(): Promise<void> {
    console.log('📁 Loading datasets...');

    await this.loadDatasetInfo(); // Load to verify it exists
    const baseDir = this.config.dataDir;

    // Load training data
    const trainData = await this.loadDatasetFile(join(baseDir, 'pytorch', 'train_regression.json'));
    const valData = await this.loadDatasetFile(
      join(baseDir, 'pytorch', 'validation_regression.json'),
    );

    console.log(`  Training samples: ${trainData.samples.length}`);
    console.log(`  Validation samples: ${valData.samples.length}`);

    // Create TensorFlow datasets
    this.trainDataset = this.createTFDataset(trainData.samples, baseDir, true) as any;
    this.valDataset = this.createTFDataset(valData.samples, baseDir, false) as any;
  }

  private async loadDatasetInfo(): Promise<any> {
    const infoPath = join(this.config.dataDir, 'dataset_info.json');
    const content = await fs.readFile(infoPath, 'utf8');
    return JSON.parse(content);
  }

  private async loadDatasetFile(filePath: string): Promise<any> {
    const content = await fs.readFile(filePath, 'utf8');
    return JSON.parse(content);
  }

  private createTFDataset(
    samples: DatasetSample[],
    baseDir: string,
    isTraining: boolean,
  ) {
    console.log(
      `📊 Creating ${isTraining ? 'training' : 'validation'} dataset with ${samples.length} samples`,
    );

    // Create dataset from generator
    const dataset = tf.data.generator(async function* () {
      let successCount = 0;
      let errorCount = 0;

      for (let i = 0; i < samples.length; i++) {
        const sample = samples[i];

        try {
          // Load and preprocess image
          const imagePath = path.isAbsolute(sample?.screenshot || '')
            ? sample?.screenshot
            : path.join(baseDir, sample?.screenshot || '');

          // Check if file exists
          try {
            await fs.access(imagePath || '');
          } catch {
            console.warn(`⚠️  Image not found: ${imagePath}`);
            errorCount++;
            continue;
          }

          const imageBuffer = await fs.readFile(imagePath || '');

          // Decode image using TensorFlow.js
          let imageTensor: tf.Tensor3D;
          try {
            imageTensor = tf.node.decodeImage(imageBuffer, 3) as tf.Tensor3D;
          } catch (decodeError) {
            console.warn(
              `⚠️  Failed to decode image ${path.basename(imagePath || '')}: ${decodeError}`,
            );
            errorCount++;
            continue;
          }

          // Resize to model input size (240x320)
          const resized = tf.image.resizeBilinear(imageTensor, [240, 320]);
          imageTensor.dispose();

          // Normalize to [0, 1] and apply augmentation if training
          let processed = tf.div(resized, 255.0);
          resized.dispose();

          if (isTraining) {
            // Simple data augmentation
            const shouldFlip = Math.random() < 0.05; // 5% chance
            if (shouldFlip) {
              const flipped = tf.image.flipLeftRight(processed as tf.Tensor4D);
              processed.dispose();
              processed = flipped;
            }

            // Slight brightness adjustment
            const brightnessAdjust = (Math.random() - 0.5) * 0.1; // ±5%
            if (Math.abs(brightnessAdjust) > 0.01) {
              const brightened = tf.add(processed, brightnessAdjust);
              processed.dispose();
              processed = brightened;
            }
          }

          // Create target tensor with default values for undefined
          const target = tf.tensor1d([
            sample?.outputs.movement_x ?? 0,
            sample?.outputs.movement_y ?? 0,
            sample?.outputs.aim_x ?? 0,
            sample?.outputs.aim_y ?? 0,
            sample?.outputs.shooting ?? 0,
          ]);

          successCount++;

          // Progress logging
          if (successCount % 50 === 0) {
            console.log(
              `  📸 Processed ${successCount}/${samples.length} images (${errorCount} errors)`,
            );
          }

          yield { xs: processed, ys: target };
        } catch (error) {
          console.warn(`⚠️  Error processing sample ${i}: ${error}`);
          errorCount++;
          continue;
        }
      }

      console.log(`✅ Dataset creation complete: ${successCount} successful, ${errorCount} errors`);
    });

    // Apply batching and shuffling
    let processedDataset = dataset;

    if (isTraining) {
      processedDataset = processedDataset.shuffle(1000);
    }

    return processedDataset.batch(this.config.batchSize);
  }

  private async createModel(): Promise<tf.LayersModel> {
    console.log(`🤖 Creating ${this.config.modelType} model...`);

    switch (this.config.modelType) {
      case 'custom_cnn':
        return this.createCustomCNN();
      case 'mobilenet':
        return await this.createMobileNetModel();
      case 'efficientnet':
        return await this.createEfficientNetModel();
      default:
        throw new Error(`Unknown model type: ${this.config.modelType}`);
    }
  }

  private createCustomCNN(): tf.LayersModel {
    console.log('🧠 Creating custom CNN optimized for Nuclear Throne...');

    const model = tf.sequential({
      layers: [
        // Input layer - Nuclear Throne resolution
        tf.layers.conv2d({
          inputShape: [240, 320, 3],
          filters: 32,
          kernelSize: 3,
          activation: 'relu',
          padding: 'same',
          name: 'conv1',
        }),
        tf.layers.batchNormalization({ name: 'bn1' }),
        tf.layers.maxPooling2d({ poolSize: 2, name: 'pool1' }), // 120x160

        // Feature extraction layers
        tf.layers.conv2d({
          filters: 64,
          kernelSize: 3,
          activation: 'relu',
          padding: 'same',
          name: 'conv2',
        }),
        tf.layers.batchNormalization({ name: 'bn2' }),
        tf.layers.maxPooling2d({ poolSize: 2, name: 'pool2' }), // 60x80

        tf.layers.conv2d({
          filters: 128,
          kernelSize: 3,
          activation: 'relu',
          padding: 'same',
          name: 'conv3',
        }),
        tf.layers.batchNormalization({ name: 'bn3' }),
        tf.layers.maxPooling2d({ poolSize: 2, name: 'pool3' }), // 30x40

        tf.layers.conv2d({
          filters: 256,
          kernelSize: 3,
          activation: 'relu',
          padding: 'same',
          name: 'conv4',
        }),
        tf.layers.batchNormalization({ name: 'bn4' }),
        tf.layers.globalAveragePooling2d({ name: 'gap' }),

        // Decision layers
        tf.layers.dropout({ rate: 0.3, name: 'dropout1' }),
        tf.layers.dense({
          units: 256,
          activation: 'relu',
          kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }),
          name: 'dense1',
        }),
        tf.layers.dropout({ rate: 0.2, name: 'dropout2' }),
        tf.layers.dense({
          units: 64,
          activation: 'relu',
          name: 'dense2',
        }),

        // Output layer - 5 outputs for game actions
        tf.layers.dense({
          units: 5,
          activation: 'linear',
          name: 'output',
        }),
      ],
    });

    return model;
  }

  private async createMobileNetModel(): Promise<tf.LayersModel> {
    // Load MobileNet base (this is a simplified version - TensorFlow.js has limited pre-trained models)
    const baseModel = tf.sequential({
      layers: [
        tf.layers.conv2d({
          inputShape: [240, 320, 3],
          filters: 32,
          kernelSize: 3,
          strides: 2,
          activation: 'relu',
          padding: 'same',
        }),
        tf.layers.depthwiseConv2d({ kernelSize: 3, activation: 'relu', padding: 'same' }),
        tf.layers.conv2d({ filters: 64, kernelSize: 1, activation: 'relu' }),
        tf.layers.maxPooling2d({ poolSize: 2 }),

        tf.layers.depthwiseConv2d({ kernelSize: 3, activation: 'relu', padding: 'same' }),
        tf.layers.conv2d({ filters: 128, kernelSize: 1, activation: 'relu' }),
        tf.layers.maxPooling2d({ poolSize: 2 }),

        tf.layers.depthwiseConv2d({ kernelSize: 3, activation: 'relu', padding: 'same' }),
        tf.layers.conv2d({ filters: 256, kernelSize: 1, activation: 'relu' }),
        tf.layers.globalAveragePooling2d({ dataFormat: 'channelsLast' }),

        // Custom head
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 128, activation: 'relu' }),
        tf.layers.dense({ units: 5, activation: 'linear' }),
      ],
    });

    return baseModel;
  }

  private async createEfficientNetModel(): Promise<tf.LayersModel> {
    // Simplified EfficientNet-like architecture
    const model = tf.sequential({
      layers: [
        // Stem
        tf.layers.conv2d({
          inputShape: [240, 320, 3],
          filters: 32,
          kernelSize: 3,
          strides: 2,
          activation: 'swish',
          padding: 'same',
        }),

        // Mobile inverted bottleneck blocks (simplified)
        tf.layers.separableConv2d({
          filters: 64,
          kernelSize: 3,
          activation: 'swish',
          padding: 'same',
        }),
        tf.layers.maxPooling2d({ poolSize: 2 }),

        tf.layers.separableConv2d({
          filters: 128,
          kernelSize: 3,
          activation: 'swish',
          padding: 'same',
        }),
        tf.layers.maxPooling2d({ poolSize: 2 }),

        tf.layers.separableConv2d({
          filters: 256,
          kernelSize: 3,
          activation: 'swish',
          padding: 'same',
        }),
        tf.layers.globalAveragePooling2d({ dataFormat: 'channelsLast' }),

        // Head
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 256, activation: 'swish' }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 64, activation: 'swish' }),
        tf.layers.dense({ units: 5, activation: 'linear' }),
      ],
    });

    return model;
  }

  private createCustomLoss() {
    return (yTrue: tf.Tensor, yPred: tf.Tensor) => {
      return tf.tidy(() => {
        // Split predictions and targets
        const predMovement = tf.slice(yPred, [0, 0], [-1, 2]);
        const predAim = tf.slice(yPred, [0, 2], [-1, 2]);
        const predShoot = tf.slice(yPred, [0, 4], [-1, 1]);

        const trueMovement = tf.slice(yTrue, [0, 0], [-1, 2]);
        const trueAim = tf.slice(yTrue, [0, 2], [-1, 2]);
        const trueShoot = tf.slice(yTrue, [0, 4], [-1, 1]);

        // Different loss components
        const movementLoss = tf.losses.meanSquaredError(trueMovement, predMovement);
        const aimLoss = tf.losses.meanSquaredError(trueAim, predAim);
        const shootLoss = tf.losses.sigmoidCrossEntropy(trueShoot, predShoot);

        // Weighted combination
        const totalLoss = tf.add(
          tf.add(tf.mul(movementLoss, 2.0), tf.mul(aimLoss, 1.5)),
          tf.mul(shootLoss, 1.0),
        );

        return totalLoss;
      });
    };
  }

  async train(): Promise<void> {
    if (!this.model || !this.trainDataset || !this.valDataset) {
      throw new Error('Model or datasets not initialized');
    }

    console.log('🚀 Starting training...');

    // Compile model
    this.model.compile({
      optimizer: tf.train.adam(this.config.learningRate),
      loss: this.createCustomLoss(),
      metrics: ['mse'],
    });

    // Print model summary
    this.model.summary();

    // Create callbacks
    const self = this;
    const callbacks = [
      new (class extends tf.Callback {
        override async onEpochEnd(epoch: number, logs?: tf.Logs) {
          console.log(
            `Epoch ${epoch + 1}: loss=${logs?.['loss']?.toFixed(4)}, val_loss=${logs?.['val_loss']?.toFixed(4)}`,
          );

          // Save model checkpoint
          if (logs?.['val_loss'] && (epoch === 0 || logs['val_loss'] < self.getBestValLoss())) {
            await self.saveModel(epoch, logs['val_loss'] as number);
          }
        }
        override async onTrainEnd() {
          console.log('✅ Training completed!');
        }
      })(),
    ];

    // Train the model
    const history = await this.model.fitDataset(this.trainDataset, {
      epochs: this.config.epochs,
      validationData: this.valDataset,
      callbacks,
    });

    console.log('📊 Training history:', history.history);
  }

  private bestValLoss = Infinity;

  private getBestValLoss(): number {
    return this.bestValLoss;
  }

  private async saveModel(epoch: number, valLoss: number): Promise<void> {
    if (!this.model) return;

    this.bestValLoss = valLoss;

    // Save model
    const modelPath = `file://${path.resolve(this.config.savePath)}`;
    await this.model.save(modelPath);

    // Save metadata
    const metadata = {
      epoch,
      valLoss,
      config: this.config,
      savedAt: new Date().toISOString(),
    };

    const metadataPath = this.config.savePath.replace('.json', '_metadata.json');
    await fs.writeFile(metadataPath, JSON.stringify(metadata, null, 2));

    console.log(`💾 Model saved: epoch ${epoch + 1}, val_loss=${valLoss.toFixed(4)}`);
  }

  async loadModel(modelPath: string): Promise<void> {
    console.log(`📂 Loading model from: ${modelPath}`);
    this.model = await tf.loadLayersModel(`file://${path.resolve(modelPath)}`);
    console.log('✅ Model loaded successfully');
  }

  // Inference method
  async predict(imagePath: string): Promise<{
    movement: { x: number; y: number };
    aim: { x: number; y: number };
    shooting: number;
  }> {
    if (!this.model) {
      throw new Error('Model not loaded');
    }

    // Load and preprocess image
    const imageBuffer = await fs.readFile(imagePath);
    let imageTensor = tf.node.decodeImage(imageBuffer, 3) as tf.Tensor3D;
    imageTensor = tf.image.resizeBilinear(imageTensor, [240, 320]);
    imageTensor = tf.div(imageTensor, 255.0);

    // Add batch dimension
    const batchedImage = tf.expandDims(imageTensor, 0);

    // Predict
    const prediction = this.model.predict(batchedImage) as tf.Tensor;
    const predictionData = await prediction.data();

    // Clean up tensors
    imageTensor.dispose();
    batchedImage.dispose();
    prediction.dispose();

    return {
      movement: {
        x: predictionData[0] ?? 0,
        y: predictionData[1] ?? 0,
      },
      aim: {
        x: predictionData[2] ?? 0,
        y: predictionData[3] ?? 0,
      },
      shooting: predictionData[4] ?? 0,
    };
  }

  dispose(): void {
    if (this.model) {
      this.model.dispose();
    }
  }
}

// CLI interface
async function main() {
  const args = process.argv.slice(2);

  if (args.length < 1) {
    console.log('Usage: ts-node tfjs-training-setup.ts <data-dir> [options]');
    console.log('');
    console.log('Options:');
    console.log(
      '  --model <type>          Model type: custom_cnn, mobilenet, efficientnet (default: custom_cnn)',
    );
    console.log('  --epochs <num>          Number of epochs (default: 50)');
    console.log('  --batch-size <num>      Batch size (default: 16)');
    console.log('  --learning-rate <num>   Learning rate (default: 0.001)');
    console.log('  --save-path <path>      Model save path (default: ./model)');
    console.log('  --resume <path>         Resume from saved model');
    console.log('');
    console.log('Example:');
    console.log('  ts-node tfjs-training-setup.ts ./cleaned_data --model mobilenet --epochs 30');
    process.exit(1);
  }

  const config: TrainingConfig = {
    dataDir: args[0] || '',
    modelType: 'custom_cnn',
    epochs: 50,
    batchSize: 16,
    learningRate: 0.001,
    validationSplit: 0.2,
    savePath: './model',
  };

  // Parse options
  for (let i = 1; i < args.length; i++) {
    switch (args[i]) {
      case '--model':
        config.modelType = args[++i] as any;
        break;
      case '--epochs':
        config.epochs = parseInt(args[++i] || '');
        break;
      case '--batch-size':
        config.batchSize = parseInt(args[++i] || '');
        break;
      case '--learning-rate':
        config.learningRate = parseFloat(args[++i] || '');
        break;
      case '--save-path':
        config.savePath = args[++i] || '';
        break;
      case '--resume':
        config.resumeFrom = args[++i];
        break;
    }
  }

  console.log('📋 Training configuration:', config);

  try {
    const trainer = new TensorFlowTrainer(config);

    if (config.resumeFrom) {
      await trainer.loadModel(config.resumeFrom);
    }

    await trainer.initialize();
    await trainer.train();

    trainer.dispose();
  } catch (error) {
    console.error('❌ Training failed:', error);
    process.exit(1);
  }
}

// Export for use as module
export { TensorFlowTrainer, TrainingConfig };

// Run CLI if called directly
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main();
}
