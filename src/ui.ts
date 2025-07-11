import blessed from 'blessed';
import { Widgets } from 'blessed';
import type { Mode, ApplicationState } from './index.js';

export class NuclearThroneUI {
  private screen: Widgets.Screen;
  private modeBox: Widgets.BoxElement;
  private menuBox: Widgets.ListElement;
  private statusBox: Widgets.BoxElement;
  private rightPanel: Widgets.BoxElement | null = null;
  private state: ApplicationState;
  private onModeChange: (mode: Mode) => void;
  private modes: { value: Mode; label: string }[] = [
    { value: 'welcome', label: 'Welcome' },
    { value: 'data-collection', label: 'Data Collection Mode' },
    { value: 'training', label: 'Training Mode' },
    { value: 'agent', label: 'Agent Mode' },
  ];

  constructor(state: ApplicationState, onModeChange: (mode: Mode) => void) {
    this.state = state;
    this.onModeChange = onModeChange;

    this.screen = blessed.screen({
      smartCSR: true,
      title: 'Nuclear Throne Bot - Redux',
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
        type: 'line',
      },
      style: {
        border: {
          fg: 'cyan',
        },
      },
      label: ' NTB-Redux ',
    });

    this.menuBox = blessed.list({
      parent: this.modeBox,
      top: 1,
      left: 1,
      width: '30%',
      height: this.modes.length + 2, // Height based on number of items + borders
      border: {
        type: 'line',
      },
      style: {
        border: {
          fg: 'yellow',
        },
        selected: {
          bg: 'blue',
          fg: 'white',
          bold: true,
        },
      },
      label: ' Mode Selection ',
      keys: true,
      vi: true,
      mouse: true,
      items: this.modes.map((m) => m.label),
    });

    this.statusBox = blessed.box({
      bottom: 0,
      left: 0,
      width: '100%',
      height: 4,
      border: {
        type: 'line',
      },
      style: {
        border: {
          fg: 'green',
        },
      },
      content: this.getStatusText(),
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
        this.onModeChange(mode.value);
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
        type: 'line',
      },
      style: {
        border: {
          fg: 'white',
        },
      },
      label: ` ${this.modes.find((m) => m.value === this.state.currentMode)?.label || ''} `,
      content: this.getModeContent(),
      align: 'center',
      valign: 'middle',
      tags: true,
    });

    this.render();
  }

  private getModeContent(): string {
    if (this.state.currentMode === 'welcome') {
      const logo = [
        '{bold}{yellow-fg}███╗   ██╗████████╗██████╗       ██████╗ {/}',
        '{bold}{yellow-fg}████╗  ██║╚══██╔══╝██╔══██╗      ██╔══██╗{/}',
        '{bold}{yellow-fg}██╔██╗ ██║   ██║   ██████╔╝█████╗██████╔╝{/}',
        '{bold}{yellow-fg}██║╚██╗██║   ██║   ██╔══██╗╚════╝██╔══██╗{/}',
        '{bold}{yellow-fg}██║ ╚████║   ██║   ██████╔╝      ██║  ██║{/}',
        '{bold}{yellow-fg}╚═╝  ╚═══╝   ╚═╝   ╚═════╝       ╚═╝  ╚═╝{/}',
      ].join('\n');

      const welcome =
        '\n\n{bold}{green-fg}Welcome to Nuclear Throne Bot Redux!{/}\n\n' +
        '{white-fg}An AI-powered bot for playing Nuclear Throne{/}\n\n' +
        '{cyan-fg}Getting Started:{/}\n' +
        '• Select {yellow-fg}Data Collection Mode{/} to record gameplay\n' +
        '• Use {yellow-fg}Training Mode{/} to train the AI model\n' +
        '• Run {yellow-fg}Agent Mode{/} to watch the bot play\n\n' +
        '{dim-fg}Navigate with arrow keys • Press Enter to select{/}';

      return logo + welcome;
    }
    return '';
  }

  private getStatusText(): string {
    const currentModeLabel =
      this.modes.find((m) => m.value === this.state.currentMode)?.label || '';
    return (
      ` Current Mode: ${currentModeLabel}\n` +
      ` Press 'q' or 'ESC' to quit | Use arrow keys to navigate`
    );
  }

  private updateStatus(): void {
    this.statusBox.setContent(this.getStatusText());
    this.render();
  }

  private render(): void {
    this.screen.render();
  }

  public updateState(state: ApplicationState): void {
    this.state = state;
    this.updateStatus();
    this.handleModeChange();
  }

  public start(): void {
    this.handleModeChange();
  }
}