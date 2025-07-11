import { NuclearThroneUI } from './ui.js';

export type Mode = 'welcome' | 'data-collection' | 'training' | 'agent';

export interface DataCollectionConfig {
  targetWindowName: string;
  trainingDataSaveDirectory: string;
  trainingDataSetName: string;
  isDataCollectionEnabled: boolean;
}

export interface TrainingConfig {
  batchSize: number; // placeholder
  learningRate: number; // placeholder
  epochs: number; // placeholder
  modelPath: string; // placeholder
}

export interface AgentConfig {
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
  agent: AgentConfig;
}

class ApplicationStateManager {
  private state: ApplicationState;
  private ui: NuclearThroneUI;

  constructor() {
    this.state = {
      currentMode: 'welcome',
      welcome: {},
      dataCollection: {
        targetWindowName: 'Nuclear Throne',
        trainingDataSaveDirectory: './training-data',
        trainingDataSetName: 'dataset-1',
        isDataCollectionEnabled: false,
      },
      training: {
        batchSize: 32,
        learningRate: 0.001,
        epochs: 100,
        modelPath: './models',
      },
      agent: {
        inferenceDelay: 50,
        confidenceThreshold: 0.8,
        actionTimeout: 1000,
      },
    };

    this.ui = new NuclearThroneUI(
      this.state, 
      (mode) => this.handleModeChange(mode),
      (updates) => this.handleStateUpdate(updates)
    );
  }

  private handleModeChange(mode: Mode): void {
    this.state.currentMode = mode;
    this.ui.updateState(this.state);
  }

  private handleStateUpdate(updates: Partial<ApplicationState>): void {
    this.state = { ...this.state, ...updates };
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