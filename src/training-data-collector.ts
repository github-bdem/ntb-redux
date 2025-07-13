import { ScreenshotCapture } from './screenshot-capture.js';
import { InputCapture, InputEvent, SimpleInputCapture } from './input-capture.js';
import { promises as fs } from 'fs';
import { join } from 'path';

export interface TrainingDataPoint {
  timestamp: number;
  screenshot: {
    buffer?: Buffer;
    filePath?: string;
    width: number;
    height: number;
  };
  inputEvents: InputEvent[];
  gameState?: any; // Optional game state information
}

export interface DataCollectionConfig {
  outputDir: string;
  captureIntervalMs: number;
  saveScreenshotsToFiles: boolean;
  gameWindowTitle: string;
  maxDataPoints: number;
}

export class TrainingDataCollector {
  private screenshotCapture: ScreenshotCapture;
  private inputCapture: SimpleInputCapture; // Using simple version for reliability
  private gameWindowId: string | null = null;
  private isCollecting = false;
  private dataPoints: TrainingDataPoint[] = [];
  private config: DataCollectionConfig;

  constructor(config: DataCollectionConfig) {
    this.screenshotCapture = new ScreenshotCapture();
    this.inputCapture = new SimpleInputCapture();
    this.config = config;
  }

  async initialize(): Promise<void> {
    // Ensure output directory exists
    await fs.mkdir(this.config.outputDir, { recursive: true });

    // Find game window
    await this.findGameWindow();

    console.log(`Initialized data collector for window ID: ${this.gameWindowId}`);
  }

  private async findGameWindow(): Promise<void> {
    const windows = await this.screenshotCapture.getWindows();
    const gameWindow = windows.find((w) =>
      w.title.toLowerCase().includes(this.config.gameWindowTitle.toLowerCase()),
    );

    if (!gameWindow) {
      throw new Error(`Game window not found: ${this.config.gameWindowTitle}`);
    }

    this.gameWindowId = gameWindow.id;
    console.log(`Found game window: ${gameWindow.title} (ID: ${gameWindow.id})`);
  }

  async startCollection(): Promise<void> {
    if (this.isCollecting) {
      throw new Error('Data collection already in progress');
    }

    if (!this.gameWindowId) {
      await this.initialize();
    }

    this.isCollecting = true;
    this.dataPoints = [];

    // Start input capture
    await this.inputCapture.startCapturing();

    console.log(`Starting data collection with ${this.config.captureIntervalMs}ms intervals...`);

    // Main collection loop
    const interval = setInterval(async () => {
      if (!this.isCollecting) {
        clearInterval(interval);
        return;
      }

      try {
        await this.collectDataPoint();

        // Check if we've reached max data points
        if (this.dataPoints.length >= this.config.maxDataPoints) {
          console.log(
            `Reached maximum data points (${this.config.maxDataPoints}), stopping collection`,
          );
          await this.stopCollection();
        }
      } catch (error) {
        console.error('Error collecting data point:', error);
      }
    }, this.config.captureIntervalMs);

    // Store interval reference for cleanup
    (this as any).collectionInterval = interval;
  }

  private async collectDataPoint(): Promise<void> {
    const timestamp = Date.now();

    // Get recent input events (this line would be for the full not simple version)
    // const inputEvents = this.inputCapture.getNewEvents ? this.inputCapture.getNewEvents() : [];
    const inputEvents = [] as InputEvent[];

    // Capture screenshot
    let screenshot: TrainingDataPoint['screenshot'];

    if (this.config.saveScreenshotsToFiles) {
      const filename = `screenshot_${timestamp}.png`;
      const filePath = join(this.config.outputDir, filename);

      await this.screenshotCapture.captureWindowById(this.gameWindowId!, filePath);

      // Get image dimensions (assume 320x240 for now, could be dynamic)
      screenshot = {
        filePath,
        width: 320,
        height: 240,
      };
    } else {
      const buffer = await this.screenshotCapture.captureWindowToBuffer(this.gameWindowId!);
      screenshot = {
        buffer,
        width: 320,
        height: 240,
      };
    }

    const dataPoint: TrainingDataPoint = {
      timestamp,
      screenshot,
      inputEvents,
    };

    this.dataPoints.push(dataPoint);

    console.log(
      `Collected data point ${this.dataPoints.length}/${this.config.maxDataPoints} - ${inputEvents.length} input events`,
    );
  }

  async stopCollection(): Promise<TrainingDataPoint[]> {
    if (!this.isCollecting) {
      return this.dataPoints;
    }

    this.isCollecting = false;

    // Stop input capture
    this.inputCapture.stopCapturing();

    // Clear interval if it exists
    if ((this as any).collectionInterval) {
      clearInterval((this as any).collectionInterval);
    }

    console.log(`Data collection stopped. Collected ${this.dataPoints.length} data points.`);

    // Save metadata
    await this.saveCollectionMetadata();

    return this.dataPoints;
  }

  private async saveCollectionMetadata(): Promise<void> {
    const metadata = {
      collectionInfo: {
        timestamp: Date.now(),
        gameWindowTitle: this.config.gameWindowTitle,
        captureIntervalMs: this.config.captureIntervalMs,
        totalDataPoints: this.dataPoints.length,
        config: this.config,
      },
      dataPoints: this.dataPoints.map((dp) => ({
        timestamp: dp.timestamp,
        screenshotFile: dp.screenshot.filePath,
        screenshotSize: dp.screenshot.buffer?.length,
        inputEventCount: dp.inputEvents.length,
        inputEvents: dp.inputEvents,
      })),
    };

    const metadataPath = join(this.config.outputDir, 'collection_metadata.json');
    await fs.writeFile(metadataPath, JSON.stringify(metadata, null, 2));

    console.log(`Saved collection metadata to: ${metadataPath}`);
  }

  // Get current data points without stopping collection
  getCurrentData(): TrainingDataPoint[] {
    return [...this.dataPoints];
  }

  // Export data in various formats
  async exportToFormat(format: 'json' | 'csv'): Promise<string> {
    const exportPath = join(this.config.outputDir, `training_data.${format}`);

    if (format === 'json') {
      await fs.writeFile(exportPath, JSON.stringify(this.dataPoints, null, 2));
    } else if (format === 'csv') {
      // Create CSV with flattened data
      const csvLines = ['timestamp,screenshot_path,input_events_json'];

      for (const dp of this.dataPoints) {
        const line = [
          dp.timestamp,
          dp.screenshot.filePath || 'buffer',
          JSON.stringify(dp.inputEvents).replace(/,/g, ';'), // Escape commas in JSON
        ].join(',');
        csvLines.push(line);
      }

      await fs.writeFile(exportPath, csvLines.join('\n'));
    }

    console.log(`Exported training data to: ${exportPath}`);
    return exportPath;
  }
}
