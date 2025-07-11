import blessed from 'blessed';
import { Widgets } from 'blessed';

type Mode = 'data-collection' | 'training' | 'live';

interface DataCollectionConfig {
  captureInterval: number; // placeholder
  saveDirectory: string; // placeholder
  maxSamples: number; // placeholder
}

interface TrainingConfig {
  batchSize: number; // placeholder
  learningRate: number; // placeholder
  epochs: number; // placeholder
  modelPath: string; // placeholder
}

interface LiveConfig {
  inferenceDelay: number; // placeholder
  confidenceThreshold: number; // placeholder
  actionTimeout: number; // placeholder
}

interface ApplicationState {
  currentMode: Mode;
  dataCollection: DataCollectionConfig;
  training: TrainingConfig;
  live: LiveConfig;
}

const initialState: ApplicationState = {
  currentMode: 'data-collection',
  dataCollection: {
    captureInterval: 100,
    saveDirectory: './data',
    maxSamples: 10000
  },
  training: {
    batchSize: 32,
    learningRate: 0.001,
    epochs: 100,
    modelPath: './models'
  },
  live: {
    inferenceDelay: 50,
    confidenceThreshold: 0.8,
    actionTimeout: 1000
  }
};

class NuclearThroneUI {
  private screen: Widgets.Screen;
  private modeBox: Widgets.BoxElement;
  private menuBox: Widgets.ListElement;
  private statusBox: Widgets.BoxElement;
  private rightPanel: Widgets.BoxElement | null = null;
  private state: ApplicationState;
  private modes: { value: Mode; label: string }[] = [
    { value: 'data-collection', label: 'Data Collection Mode' },
    { value: 'training', label: 'Training Mode' },
    { value: 'live', label: 'Live Mode' }
  ];

  constructor() {
    this.state = { ...initialState };
    
    this.screen = blessed.screen({
      smartCSR: true,
      title: 'Nuclear Throne Bot - Redux'
    });

    this.screen.key(['escape', 'q', 'C-c'], () => {
      process.exit(0);
    });

    this.modeBox = blessed.box({
      top: 0,
      left: 0,
      width: '100%',
      height: '100%-4',
      border: {
        type: 'line'
      },
      style: {
        border: {
          fg: 'cyan'
        }
      },
      label: ' NTB-Redux '
    });

    this.menuBox = blessed.list({
      parent: this.modeBox,
      top: 1,
      left: 1,
      width: '30%',
      height: this.modes.length + 2, // Height based on number of items + borders
      border: {
        type: 'line'
      },
      style: {
        border: {
          fg: 'yellow'
        },
        selected: {
          bg: 'blue',
          fg: 'white',
          bold: true
        }
      },
      label: ' Mode Selection ',
      keys: true,
      vi: true,
      mouse: true,
      items: this.modes.map(m => m.label)
    });

    this.statusBox = blessed.box({
      bottom: 0,
      left: 0,
      width: '100%',
      height: 4,
      border: {
        type: 'line'
      },
      style: {
        border: {
          fg: 'green'
        }
      },
      content: this.getStatusText()
    });

    this.screen.append(this.modeBox);
    this.screen.append(this.statusBox);

    this.setupEventHandlers();
    this.render();
  }

  private setupEventHandlers(): void {
    this.menuBox.on('select', (_item: any, index: number) => {
      const mode = this.modes[index];
      if (mode) {
        this.state.currentMode = mode.value;
        this.updateStatus();
        this.handleModeChange();
      }
    });

    this.menuBox.focus();
  }

  private handleModeChange(): void {
    // Remove old right panel if exists
    if (this.rightPanel) {
      this.rightPanel.destroy();
    }

    // Create new right panel
    this.rightPanel = blessed.box({
      parent: this.modeBox,
      top: 1,
      left: '31%',
      width: '68%',
      height: '100%-2',
      border: {
        type: 'line'
      },
      style: {
        border: {
          fg: 'white'
        }
      },
      label: ` ${this.modes.find(m => m.value === this.state.currentMode)?.label || ''} `
    });

    this.render();
  }


  private getStatusText(): string {
    const currentModeLabel = this.modes.find(m => m.value === this.state.currentMode)?.label || '';
    return ` Current Mode: ${currentModeLabel}\n` +
           ` Press 'q' or 'ESC' to quit | Use arrow keys to navigate`;
  }

  private updateStatus(): void {
    this.statusBox.setContent(this.getStatusText());
    this.render();
  }

  private render(): void {
    this.screen.render();
  }

  public start(): void {
    this.handleModeChange();
  }
}

export function main(): void {
  const ui = new NuclearThroneUI();
  ui.start();
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}