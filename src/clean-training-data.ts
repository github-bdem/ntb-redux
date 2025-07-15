#!/usr/bin/env ts-node

import { promises as fs } from 'fs';
import { join, basename } from 'path';
import { fileURLToPath } from 'url';
import { GameDataPreprocessor } from './data-preprocessor.js';

interface CleaningConfig {
  inputDir: string;
  outputDir: string;
  validationSplit: number; // Percentage for validation (e.g., 0.2 for 20%)
  testSplit: number; // Percentage for test (e.g., 0.1 for 10%)
  minInputEvents: number; // Minimum input events per frame to keep
  maxMouseMoveDistance: number; // Filter out large mouse jumps (pixels)
}

class TrainingDataCleaner {
  private preprocessor: GameDataPreprocessor;
  private config: CleaningConfig;

  constructor(config: CleaningConfig) {
    this.preprocessor = new GameDataPreprocessor();
    this.config = config;
  }

  async cleanEntireDirectory(): Promise<void> {
    console.log('🧹 Starting training data cleaning process...');
    console.log(`Input directory: ${this.config.inputDir}`);
    console.log(`Output directory: ${this.config.outputDir}`);

    // Ensure output directory exists
    await fs.mkdir(this.config.outputDir, { recursive: true });
    await fs.mkdir(join(this.config.outputDir, 'pytorch'), { recursive: true });
    await fs.mkdir(join(this.config.outputDir, 'screenshots'), { recursive: true });

    // Find all training sessions
    const sessions = await this.findTrainingSessions();
    console.log(`Found ${sessions.length} training sessions`);

    if (sessions.length === 0) {
      throw new Error(
        'No training sessions found! Make sure collection_metadata.json files exist.',
      );
    }

    // Process all sessions
    const allProcessedData = [];
    let totalFrames = 0;
    let validFrames = 0;

    for (const session of sessions) {
      console.log(`\n📁 Processing session: ${basename(session.dir)}`);

      try {
        const processedData = await this.preprocessor.processTrainingSession(session.metadataPath);
        const filteredData = this.filterLowQualityFrames(processedData);

        console.log(`  Raw frames: ${processedData.length}`);
        console.log(`  Valid frames: ${filteredData.length}`);
        console.log(`  Filtered out: ${processedData.length - filteredData.length}`);

        // Copy screenshots to centralized location
        await this.copyScreenshots(filteredData);

        allProcessedData.push(...filteredData);
        totalFrames += processedData.length;
        validFrames += filteredData.length;
      } catch (error) {
        console.error(`  ❌ Error processing session: ${error}`);
        continue;
      }
    }

    console.log(`\n📊 Overall Statistics:`);
    console.log(`  Total frames processed: ${totalFrames}`);
    console.log(`  Valid frames kept: ${validFrames}`);
    console.log(
      `  Filter rate: ${(((totalFrames - validFrames) / totalFrames) * 100).toFixed(1)}%`,
    );

    if (validFrames === 0) {
      throw new Error('No valid frames after filtering! Check your filtering criteria.');
    }

    // Split data into train/val/test
    const splits = this.splitData(allProcessedData);
    console.log(`\n📂 Data splits:`);
    console.log(`  Training: ${splits.train.length} frames`);
    console.log(`  Validation: ${splits.validation.length} frames`);
    console.log(`  Test: ${splits.test.length} frames`);

    // Create training data in different formats
    await this.createTrainingDatasets(splits);

    console.log('\n✅ Training data cleaning completed successfully!');
  }

  private async findTrainingSessions(): Promise<Array<{ dir: string; metadataPath: string }>> {
    const sessions = [];

    try {
      const entries = await fs.readdir(this.config.inputDir, { withFileTypes: true });

      for (const entry of entries) {
        if (entry.isDirectory()) {
          const sessionDir = join(this.config.inputDir, entry.name);
          const metadataPath = join(sessionDir, 'collection_metadata.json');

          try {
            await fs.access(metadataPath);
            sessions.push({ dir: sessionDir, metadataPath });
          } catch {
            console.warn(`  ⚠️  Skipping ${entry.name}: no collection_metadata.json found`);
          }
        }
      }
    } catch (error) {
      throw new Error(`Failed to read input directory: ${error}`);
    }

    return sessions;
  }

  private filterLowQualityFrames(data: any[]): any[] {
    return data.filter((frame) => {
      // Filter frames with too few input events (likely idle/static moments)
      if (frame.inputEvents && frame.inputEvents.length < this.config.minInputEvents) {
        return false;
      }

      // Filter frames with unrealistic mouse movements
      const mouseEvents =
        frame.inputEvents?.filter(
          (e: { type: string; action: string }) => e.type === 'mouse' && e.action === 'move',
        ) || [];
      for (let i = 1; i < mouseEvents.length; i++) {
        const prev = mouseEvents[i - 1];
        const curr = mouseEvents[i];
        const distance = Math.sqrt(Math.pow(curr.x - prev.x, 2) + Math.pow(curr.y - prev.y, 2));

        if (distance > this.config.maxMouseMoveDistance) {
          return false; // Likely a teleport or error
        }
      }

      // Filter frames without valid actions
      if (!frame.actions) {
        return false;
      }

      // Keep frames with meaningful actions
      const hasMovement = frame.actions.movement !== null;
      const hasShooting = frame.actions.shooting === true;
      const hasMouseAim = frame.actions.mouseAim !== null;

      return hasMovement || hasShooting || hasMouseAim;
    });
  }

  private async copyScreenshots(data: any[]): Promise<void> {
    const screenshotDir = join(this.config.outputDir, 'screenshots');

    for (const frame of data) {
      if (frame.screenshotPath) {
        const sourcePath = frame.screenshotPath;
        const filename = basename(sourcePath);
        const destPath = join(screenshotDir, filename);

        try {
          await fs.copyFile(sourcePath, destPath);
          // Update path to relative location
          frame.screenshotPath = join('screenshots', filename);
        } catch (error) {
          console.warn(`  ⚠️  Failed to copy screenshot ${filename}: ${error}`);
        }
      }
    }
  }

  private splitData(data: any[]): { train: any[]; validation: any[]; test: any[] } {
    // Shuffle data randomly
    const shuffled = [...data].sort(() => Math.random() - 0.5);

    const totalCount = shuffled.length;
    const testCount = Math.floor(totalCount * this.config.testSplit);
    const valCount = Math.floor(totalCount * this.config.validationSplit);
    const trainCount = totalCount - testCount - valCount;

    return {
      train: shuffled.slice(0, trainCount),
      validation: shuffled.slice(trainCount, trainCount + valCount),
      test: shuffled.slice(trainCount + valCount),
    };
  }

  private async createTrainingDatasets(splits: {
    train: any[];
    validation: any[];
    test: any[];
  }): Promise<void> {
    console.log('\n🔄 Creating training datasets...');

    for (const [splitName, data] of Object.entries(splits)) {
      console.log(`  Creating ${splitName} dataset...`);

      // Create regression data (recommended for your use case)
      const regressionData = await this.preprocessor.createTrainingData(data, 'regression');

      // Save PyTorch format
      await this.savePyTorchDataset(regressionData, splitName);

      // Save classification data (alternative approach)
      const classificationData = await this.preprocessor.createTrainingData(data, 'classification');
      await this.saveClassificationDataset(classificationData, splitName);
    }

    // Create dataset info file
    await this.createDatasetInfo(splits);
  }

  private async savePyTorchDataset(data: any, splitName: string): Promise<void> {
    const dataset = {
      format: 'pytorch_regression',
      split: splitName,
      image_size: [240, 320], // height, width
      channels: 3,
      output_size: 5, // movement_x, movement_y, aim_x, aim_y, shooting
      output_names: ['movement_x', 'movement_y', 'aim_x', 'aim_y', 'shooting'],
      samples: data.samples.map((sample: { screenshot: any; outputs: any; timestamp: any }) => ({
        screenshot: sample.screenshot,
        outputs: sample.outputs,
        timestamp: sample.timestamp,
      })),
      statistics: this.calculateDatasetStats(data.samples),
    };

    const outputPath = join(this.config.outputDir, 'pytorch', `${splitName}_regression.json`);
    await fs.writeFile(outputPath, JSON.stringify(dataset, null, 2));
  }

  private async saveClassificationDataset(data: any, splitName: string): Promise<void> {
    const dataset = {
      format: 'pytorch_classification',
      split: splitName,
      image_size: [240, 320],
      channels: 3,
      num_classes: data.classes.length,
      class_names: data.classes,
      samples: data.samples.map(
        (sample: { screenshot: any; actionClass: any; timestamp: any }) => ({
          screenshot: sample.screenshot,
          action_class: sample.actionClass,
          class_index: data.classes.indexOf(sample.actionClass),
          timestamp: sample.timestamp,
        }),
      ),
    };

    const outputPath = join(this.config.outputDir, 'pytorch', `${splitName}_classification.json`);
    await fs.writeFile(outputPath, JSON.stringify(dataset, null, 2));
  }

  private calculateDatasetStats(samples: any[]): any {
    if (samples.length === 0) return {};

    const movements = samples.map((s) => [s.outputs.movement_x, s.outputs.movement_y]);
    const aims = samples.map((s) => [s.outputs.aim_x, s.outputs.aim_y]);
    const shootings = samples.map((s) => s.outputs.shooting);

    return {
      sample_count: samples.length,
      movement_stats: {
        mean_x: movements.reduce((sum, m) => sum + m[0], 0) / movements.length,
        mean_y: movements.reduce((sum, m) => sum + m[1], 0) / movements.length,
        std_x: Math.sqrt(
          movements.reduce((sum, m) => sum + Math.pow(m[0] - 0, 2), 0) / movements.length,
        ),
        std_y: Math.sqrt(
          movements.reduce((sum, m) => sum + Math.pow(m[1] - 0, 2), 0) / movements.length,
        ),
      },
      aim_stats: {
        mean_x: aims.reduce((sum, a) => sum + a[0], 0) / aims.length,
        mean_y: aims.reduce((sum, a) => sum + a[1], 0) / aims.length,
      },
      shooting_rate: shootings.reduce((sum, s) => sum + s, 0) / shootings.length,
    };
  }

  private async createDatasetInfo(splits: {
    train: any[];
    validation: any[];
    test: any[];
  }): Promise<void> {
    const info = {
      created_at: new Date().toISOString(),
      source_directory: this.config.inputDir,
      cleaning_config: this.config,
      splits: {
        train: splits.train.length,
        validation: splits.validation.length,
        test: splits.test.length,
        total: splits.train.length + splits.validation.length + splits.test.length,
      },
      formats_available: ['pytorch_regression', 'pytorch_classification'],
      image_info: {
        resolution: '320x240',
        format: 'PNG',
        channels: 3,
        location: 'screenshots/',
      },
      usage_instructions: {
        pytorch_regression: 'Use *_regression.json files for continuous action prediction',
        pytorch_classification:
          'Use *_classification.json files for discrete action classification',
        screenshots:
          'All screenshots are in the screenshots/ directory with relative paths in JSON files',
      },
    };

    const infoPath = join(this.config.outputDir, 'dataset_info.json');
    await fs.writeFile(infoPath, JSON.stringify(info, null, 2));
  }
}

// CLI functionality
async function main() {
  const args = process.argv.slice(2);

  if (args.length < 2) {
    console.log('Usage: ts-node clean-training-data.ts <input_dir> <output_dir> [options]');
    console.log('');
    console.log('Options:');
    console.log('  --val-split <float>     Validation split (default: 0.2)');
    console.log('  --test-split <float>    Test split (default: 0.1)');
    console.log('  --min-events <int>      Minimum input events per frame (default: 1)');
    console.log('  --max-mouse-jump <int>  Maximum mouse jump distance in pixels (default: 200)');
    console.log('');
    console.log('Example:');
    console.log('  ts-node clean-training-data.ts ./training_data ./cleaned_data --val-split 0.25');
    process.exit(1);
  }

  if (args[0] && args[1]) {
  }

  // TODO: fix these:  `|| ''`
  const config: CleaningConfig = {
    inputDir: args[0] || '',
    outputDir: args[1] || '',
    validationSplit: 0.2,
    testSplit: 0.1,
    minInputEvents: 1,
    maxMouseMoveDistance: 200,
  };

  // Parse command line options
  for (let i = 2; i < args.length; i++) {
    switch (args[i]) {
      case '--val-split':
        config.validationSplit = parseFloat(args[++i] || '');
        break;
      case '--test-split':
        config.testSplit = parseFloat(args[++i] || '');
        break;
      case '--min-events':
        config.minInputEvents = parseInt(args[++i] || '');
        break;
      case '--max-mouse-jump':
        config.maxMouseMoveDistance = parseInt(args[++i] || '');
        break;
      default:
        console.warn(`Unknown option: ${args[i]}`);
    }
  }

  // Validate config
  if (config.validationSplit + config.testSplit >= 1.0) {
    throw new Error('Validation + test splits must be less than 1.0');
  }

  console.log('📋 Configuration:');
  console.log(`  Validation split: ${(config.validationSplit * 100).toFixed(1)}%`);
  console.log(`  Test split: ${(config.testSplit * 100).toFixed(1)}%`);
  console.log(
    `  Training split: ${((1 - config.validationSplit - config.testSplit) * 100).toFixed(1)}%`,
  );
  console.log(`  Min input events: ${config.minInputEvents}`);
  console.log(`  Max mouse jump: ${config.maxMouseMoveDistance}px`);

  try {
    const cleaner = new TrainingDataCleaner(config);
    await cleaner.cleanEntireDirectory();
  } catch (error) {
    console.error(`\n❌ Cleaning failed: ${error}`);
    process.exit(1);
  }
}

// Run if called directly
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main();
}

export { TrainingDataCleaner, CleaningConfig };
