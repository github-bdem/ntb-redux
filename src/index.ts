import { NuclearThroneUI } from './ui.js';

export type Mode = 'welcome' | 'data-collection' | 'training' | 'agent';

export interface DataCollectionConfig {
  targetWindowName: string;
  trainingDataSaveDirectory: string;
  trainingDataSetName: string;
  isDataCollectionEnabled: boolean;
}

export interface TrainingConfig {
  trainingDataDirectory: string;
  batchModeEnabled: boolean;
  targetApplicationName: string;
  modelPath: string;
  isTrainingHappening: boolean;
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
        trainingDataDirectory: './training-data',
        batchModeEnabled: false,
        targetApplicationName: 'Nuclear Throne',
        modelPath: './models/ntb-model.pt',
        isTrainingHappening: false,
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
    // Deep merge for nested objects
    if (updates.dataCollection) {
      this.state.dataCollection = { ...this.state.dataCollection, ...updates.dataCollection };
    }
    if (updates.training) {
      this.state.training = { ...this.state.training, ...updates.training };
    }
    if (updates.agent) {
      this.state.agent = { ...this.state.agent, ...updates.agent };
    }
    if (updates.currentMode !== undefined) {
      this.state.currentMode = updates.currentMode;
    }
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