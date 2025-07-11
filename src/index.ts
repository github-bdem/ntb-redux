import { NuclearThroneUI } from './ui.js';

export type Mode = 'welcome' | 'data-collection' | 'training' | 'live';

export interface DataCollectionConfig {
  captureInterval: number; // placeholder
  saveDirectory: string; // placeholder
  maxSamples: number; // placeholder
}

export interface TrainingConfig {
  batchSize: number; // placeholder
  learningRate: number; // placeholder
  epochs: number; // placeholder
  modelPath: string; // placeholder
}

export interface LiveConfig {
  inferenceDelay: number; // placeholder
  confidenceThreshold: number; // placeholder
  actionTimeout: number; // placeholder
}

export interface WelcomeConfig {
  // No config needed for welcome mode
}

export interface ApplicationState {
  currentMode: Mode;
  welcome: WelcomeConfig;
  dataCollection: DataCollectionConfig;
  training: TrainingConfig;
  live: LiveConfig;
}

class ApplicationStateManager {
  private state: ApplicationState;
  private ui: NuclearThroneUI;

  constructor() {
    this.state = {
      currentMode: 'welcome',
      welcome: {},
      dataCollection: {
        captureInterval: 100,
        saveDirectory: './data',
        maxSamples: 10000,
      },
      training: {
        batchSize: 32,
        learningRate: 0.001,
        epochs: 100,
        modelPath: './models',
      },
      live: {
        inferenceDelay: 50,
        confidenceThreshold: 0.8,
        actionTimeout: 1000,
      },
    };

    this.ui = new NuclearThroneUI(this.state, (mode) => this.handleModeChange(mode));
  }

  private handleModeChange(mode: Mode): void {
    this.state.currentMode = mode;
    this.ui.updateState(this.state);
  }

  public start(): void {
    this.ui.start();
  }
}

export function main(): void {
  const app = new ApplicationStateManager();
  app.start();
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}