import blessed from 'blessed';
import { Widgets } from 'blessed';

type Mode = 'data-collection' | 'training' | 'live';

class NuclearThroneUI {
  private screen: Widgets.Screen;
  private modeBox: Widgets.BoxElement;
  private menuBox: Widgets.ListElement;
  private statusBox: Widgets.BoxElement;
  private rightPanel: Widgets.BoxElement | null = null;
  private currentMode: Mode = 'data-collection';
  private modes: { value: Mode; label: string }[] = [
    { value: 'data-collection', label: 'Data Collection Mode' },
    { value: 'training', label: 'Training Mode' },
    { value: 'live', label: 'Live Mode' }
  ];

  constructor() {
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
      height: '100%-3',
      border: {
        type: 'line'
      },
      style: {
        border: {
          fg: 'cyan'
        }
      }
    });

    this.menuBox = blessed.list({
      parent: this.modeBox,
      top: 1,
      left: 1,
      width: '30%',
      height: '100%-2',
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
      height: 3,
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
        this.currentMode = mode.value;
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
      label: ` ${this.modes.find(m => m.value === this.currentMode)?.label || ''} `
    });

    this.render();
  }


  private getStatusText(): string {
    return ` Current Mode: ${this.modes.find(m => m.value === this.currentMode)?.label} | Press 'q' or 'ESC' to quit | Use arrow keys to navigate `;
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