import blessed from 'blessed';
import { Widgets } from 'blessed';
import type { Mode, ApplicationState } from './index.js';

export class NuclearThroneUI {
  private screen: Widgets.Screen;
  private modeBox: Widgets.BoxElement;
  private menuBox: Widgets.ListElement;
  private statusBox: Widgets.BoxElement;
  private rightPanel: Widgets.BoxElement | null = null;
  private formElements: Widgets.BlessedElement[] = [];
  private state: ApplicationState;
  private onModeChange: (mode: Mode) => void;
  private onStateUpdate: (updates: Partial<ApplicationState>) => void;
  private tabHandler: (() => void) | null = null;
  private isPanelFocused: boolean = false;
  private currentSelection: number = 0;
  private isEditingField: boolean = false;
  private updateFieldStyles: (() => void) | null = null;
  private modes: { value: Mode; label: string }[] = [
    { value: 'welcome', label: 'Welcome' },
    { value: 'data-collection', label: 'Data Collection Mode' },
    { value: 'training', label: 'Training Mode' },
    { value: 'agent', label: 'Agent Mode' },
  ];

  constructor(
    state: ApplicationState, 
    onModeChange: (mode: Mode) => void,
    onStateUpdate: (updates: Partial<ApplicationState>) => void
  ) {
    this.state = state;
    this.onModeChange = onModeChange;
    this.onStateUpdate = onStateUpdate;

    this.screen = blessed.screen({
      smartCSR: true,
      title: 'Nuclear Throne Bot - Redux',
    });

    this.screen.key(['q', 'C-c'], () => {
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
      keys: false, // Disable built-in key handling
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
    // Manual arrow key handling for menu
    this.screen.key(['up', 'down'], (_ch, key) => {
      if (!this.isPanelFocused) {
        const selected = (this.menuBox as any).selected || 0;
        if (key.name === 'up' && selected > 0) {
          this.menuBox.up(1);
          this.screen.render();
        } else if (key.name === 'down' && selected < this.modes.length - 1) {
          this.menuBox.down(1);
          this.screen.render();
        }
      }
    });

    // Enter key to select mode
    this.screen.key(['enter'], () => {
      if (!this.isPanelFocused) {
        const index = (this.menuBox as any).selected || 0;
        const mode = this.modes[index];
        if (mode) {
          this.onModeChange(mode.value);
          this.updateStatus();
          this.handleModeChange();
          if (mode.value !== 'welcome') {
            this.isPanelFocused = true;
          }
        }
      }
    });

    // ESC key to return to mode selection
    this.screen.key(['escape'], () => {
      if (this.isPanelFocused) {
        this.isPanelFocused = false;
        this.menuBox.focus();
        this.updateStatus();
        this.render();
      } else {
        // Original ESC behavior - exit app
        process.exit(0);
      }
    });

    this.menuBox.focus();
  }

  private handleModeChange(): void {
    // Clean up old key handlers
    if (this.tabHandler) {
      this.tabHandler(); // This now calls the cleanup function
      this.tabHandler = null;
    }

    // Remove old right panel and form elements if they exist
    if (this.rightPanel) {
      this.rightPanel.destroy();
    }
    this.formElements.forEach(el => el.destroy());
    this.formElements = [];

    // Reset selection only when changing modes
    this.currentSelection = 0;

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
      tags: true,
    });

    if (this.state.currentMode === 'welcome') {
      this.rightPanel.setContent(this.getModeContent());
    } else if (this.state.currentMode === 'data-collection') {
      this.createDataCollectionForm();
    } else if (this.state.currentMode === 'training') {
      this.createTrainingForm();
    }

    this.render();
  }

  private createDataCollectionForm(): void {
    if (!this.rightPanel) return;

    // Reset editing state but preserve selection if we're updating the form
    this.isEditingField = false;

    // Title
    const title = blessed.text({
      parent: this.rightPanel,
      top: 1,
      left: 2,
      content: '{bold}{cyan-fg}Data Collection Settings{/}',
      tags: true,
    });
    this.formElements.push(title);

    // Target Window Name
    const windowLabel = blessed.text({
      parent: this.rightPanel,
      top: 3,
      left: 2,
      content: 'Target Window Name:',
    });
    this.formElements.push(windowLabel);

    const windowInputWrapper = blessed.box({
      parent: this.rightPanel,
      top: 4,
      left: 2,
      width: '90%',
      height: 3,
      border: { type: 'line' },
      style: { border: { fg: this.currentSelection === 0 ? 'white' : 'gray' } },
    });
    this.formElements.push(windowInputWrapper);

    const windowInput = blessed.textbox({
      parent: windowInputWrapper,
      top: 0,
      left: 0,
      width: '100%-2',
      height: 1,
      inputOnFocus: true,
      style: { fg: 'white' },
      value: this.state.dataCollection.targetWindowName,
    });
    windowInput.on('submit', (value) => {
      this.onStateUpdate({
        dataCollection: { ...this.state.dataCollection, targetWindowName: value },
      });
      this.isEditingField = false;
      if (this.updateFieldStyles) this.updateFieldStyles();
      this.screen.render();
    });
    windowInput.on('cancel', () => {
      this.isEditingField = false;
      if (this.updateFieldStyles) this.updateFieldStyles();
      this.screen.render();
    });
    this.formElements.push(windowInput);

    // Training Data Save Directory
    const dirLabel = blessed.text({
      parent: this.rightPanel,
      top: 8,
      left: 2,
      content: 'Training Data Save Directory:',
    });
    this.formElements.push(dirLabel);

    const dirInputWrapper = blessed.box({
      parent: this.rightPanel,
      top: 9,
      left: 2,
      width: '90%',
      height: 3,
      border: { type: 'line' },
      style: { border: { fg: this.currentSelection === 1 ? 'white' : 'gray' } },
    });
    this.formElements.push(dirInputWrapper);

    const dirInput = blessed.textbox({
      parent: dirInputWrapper,
      top: 0,
      left: 0,
      width: '100%-2',
      height: 1,
      inputOnFocus: true,
      style: { fg: 'white' },
      value: this.state.dataCollection.trainingDataSaveDirectory,
    });
    dirInput.on('submit', (value) => {
      this.onStateUpdate({
        dataCollection: { ...this.state.dataCollection, trainingDataSaveDirectory: value },
      });
      this.isEditingField = false;
      if (this.updateFieldStyles) this.updateFieldStyles();
      this.screen.render();
    });
    dirInput.on('cancel', () => {
      this.isEditingField = false;
      if (this.updateFieldStyles) this.updateFieldStyles();
      this.screen.render();
    });
    this.formElements.push(dirInput);

    // Training Data Set Name
    const datasetLabel = blessed.text({
      parent: this.rightPanel,
      top: 13,
      left: 2,
      content: 'Training Data Set Name:',
    });
    this.formElements.push(datasetLabel);

    const datasetInputWrapper = blessed.box({
      parent: this.rightPanel,
      top: 14,
      left: 2,
      width: '90%',
      height: 3,
      border: { type: 'line' },
      style: { border: { fg: this.currentSelection === 2 ? 'white' : 'gray' } },
    });
    this.formElements.push(datasetInputWrapper);

    const datasetInput = blessed.textbox({
      parent: datasetInputWrapper,
      top: 0,
      left: 0,
      width: '100%-2',
      height: 1,
      inputOnFocus: true,
      style: { fg: 'white' },
      value: this.state.dataCollection.trainingDataSetName,
    });
    datasetInput.on('submit', (value) => {
      this.onStateUpdate({
        dataCollection: { ...this.state.dataCollection, trainingDataSetName: value },
      });
      this.isEditingField = false;
      if (this.updateFieldStyles) this.updateFieldStyles();
      this.screen.render();
    });
    datasetInput.on('cancel', () => {
      this.isEditingField = false;
      if (this.updateFieldStyles) this.updateFieldStyles();
      this.screen.render();
    });
    this.formElements.push(datasetInput);

    // Data Collection Enabled Checkbox
    const enabledLabel = blessed.text({
      parent: this.rightPanel,
      top: 18,
      left: 2,
      content: 'Data Collection Enabled:',
    });
    this.formElements.push(enabledLabel);

    const checkboxWrapper = blessed.box({
      parent: this.rightPanel,
      top: 18,
      left: 26,
      width: 20,
      height: 1,
      content: '',
      style: { fg: this.currentSelection === 3 ? 'white' : 'gray' },
    });
    this.formElements.push(checkboxWrapper);

    const enabledCheckbox = blessed.checkbox({
      parent: checkboxWrapper,
      top: 0,
      left: 0,
      checked: this.state.dataCollection.isDataCollectionEnabled,
      text: this.state.dataCollection.isDataCollectionEnabled ? 'Enabled ' : 'Disabled',
      style: { 
        fg: this.currentSelection === 3 ? 'white' : 'gray',
      }
    });
    enabledCheckbox.on('check', () => {
      enabledCheckbox.text = 'Enabled ';
      this.onStateUpdate({
        dataCollection: { ...this.state.dataCollection, isDataCollectionEnabled: true },
      });
      this.screen.render();
    });
    enabledCheckbox.on('uncheck', () => {
      enabledCheckbox.text = 'Disabled';
      this.onStateUpdate({
        dataCollection: { ...this.state.dataCollection, isDataCollectionEnabled: false },
      });
      this.screen.render();
    });
    this.formElements.push(enabledCheckbox);

    // Instructions
    const instructions = blessed.text({
      parent: this.rightPanel,
      bottom: 1,
      left: 2,
      content: '{gray-fg}Arrow keys to select • Enter to edit • Space to toggle checkbox • ESC for mode menu{/}',
      tags: true,
    });
    this.formElements.push(instructions);

    // Store references for navigation and activation
    const fieldWrappers = [windowInputWrapper, dirInputWrapper, datasetInputWrapper, checkboxWrapper];
    const focusableElements = [windowInput, dirInput, datasetInput, enabledCheckbox];
    
    // Initial highlighting
    this.updateFieldStyles = () => {
      fieldWrappers.forEach((wrapper, index) => {
        if (wrapper && wrapper.style) {
          // Update wrapper border for text fields
          if (index < 3) {
            wrapper.style.border = wrapper.style.border || {};
            wrapper.style.border.fg = (index === this.currentSelection && !this.isEditingField) ? 'white' : 'gray';
            if (index === this.currentSelection && this.isEditingField) {
              wrapper.style.border.fg = 'cyan';
            }
          } else {
            // Update checkbox wrapper text color
            wrapper.style.fg = (index === this.currentSelection) ? 'white' : 'gray';
          }
        }
      });
      
      // Update checkbox text color
      if (enabledCheckbox && enabledCheckbox.style) {
        enabledCheckbox.style.fg = (this.currentSelection === 3) ? 'white' : 'gray';
      }
      
      this.screen.render();
    };

    // Arrow key navigation within form
    const arrowHandler = (_ch: any, key: any) => {
      if (!this.isPanelFocused || this.isEditingField) return;
      
      if (key.name === 'up' && this.currentSelection > 0) {
        this.currentSelection--;
        if (this.updateFieldStyles) this.updateFieldStyles();
      } else if (key.name === 'down' && this.currentSelection < focusableElements.length - 1) {
        this.currentSelection++;
        if (this.updateFieldStyles) this.updateFieldStyles();
      }
    };

    // Enter key to activate field, Space only for checkbox
    const activateHandler = (_ch: any, key: any) => {
      if (!this.isPanelFocused || this.isEditingField) return;
      
      if (key.name === 'enter') {
        const element = focusableElements[this.currentSelection];
        if (element && element.parent) {
          this.isEditingField = true;
          if (this.updateFieldStyles) this.updateFieldStyles();
          
          if (this.currentSelection === 3) {
            // Handle checkbox differently
            enabledCheckbox.toggle();
            this.isEditingField = false;
            if (this.updateFieldStyles) this.updateFieldStyles();
          } else {
            element.focus();
          }
        }
      } else if (key.name === 'space' && this.currentSelection === 3) {
        // Space only works for checkbox
        enabledCheckbox.toggle();
      }
    };

    this.screen.on('keypress', arrowHandler);
    this.screen.on('keypress', activateHandler);
    
    // Store handlers for cleanup
    this.tabHandler = () => {
      this.screen.off('keypress', arrowHandler);
      this.screen.off('keypress', activateHandler);
    };

    // Initial style update
    this.updateFieldStyles();
  }

  private createTrainingForm(): void {
    if (!this.rightPanel) return;

    // Reset editing state but preserve selection if we're updating the form
    this.isEditingField = false;

    // Title
    const title = blessed.text({
      parent: this.rightPanel,
      top: 1,
      left: 2,
      content: '{bold}{cyan-fg}Training Configuration{/}',
      tags: true,
    });
    this.formElements.push(title);

    // Training Data Directory
    const dirLabel = blessed.text({
      parent: this.rightPanel,
      top: 3,
      left: 2,
      content: 'Training Data Directory:',
    });
    this.formElements.push(dirLabel);

    const dirInputWrapper = blessed.box({
      parent: this.rightPanel,
      top: 4,
      left: 2,
      width: '90%',
      height: 3,
      border: { type: 'line' },
      style: { border: { fg: this.currentSelection === 0 ? 'white' : 'gray' } },
    });
    this.formElements.push(dirInputWrapper);

    const dirInput = blessed.textbox({
      parent: dirInputWrapper,
      top: 0,
      left: 0,
      width: '100%-2',
      height: 1,
      inputOnFocus: true,
      style: { fg: 'white' },
      value: this.state.training.trainingDataDirectory,
    });
    dirInput.on('submit', (value) => {
      this.onStateUpdate({
        training: { ...this.state.training, trainingDataDirectory: value },
      });
      this.isEditingField = false;
      if (this.updateFieldStyles) this.updateFieldStyles();
      this.screen.render();
    });
    dirInput.on('cancel', () => {
      this.isEditingField = false;
      if (this.updateFieldStyles) this.updateFieldStyles();
      this.screen.render();
    });
    this.formElements.push(dirInput);

    // Batch Mode Enabled Checkbox
    const batchLabel = blessed.text({
      parent: this.rightPanel,
      top: 8,
      left: 2,
      content: 'Batch Mode Enabled:',
    });
    this.formElements.push(batchLabel);

    const batchCheckboxWrapper = blessed.box({
      parent: this.rightPanel,
      top: 8,
      left: 23,
      width: 20,
      height: 1,
      content: '',
      style: { fg: this.currentSelection === 1 ? 'white' : 'gray' },
    });
    this.formElements.push(batchCheckboxWrapper);

    const batchCheckbox = blessed.checkbox({
      parent: batchCheckboxWrapper,
      top: 0,
      left: 0,
      checked: this.state.training.batchModeEnabled,
      text: this.state.training.batchModeEnabled ? 'Enabled ' : 'Disabled',
      style: { 
        fg: this.currentSelection === 1 ? 'white' : 'gray',
      }
    });
    batchCheckbox.on('check', () => {
      batchCheckbox.text = 'Enabled ';
      this.onStateUpdate({
        training: { ...this.state.training, batchModeEnabled: true },
      });
      this.screen.render();
    });
    batchCheckbox.on('uncheck', () => {
      batchCheckbox.text = 'Disabled';
      this.onStateUpdate({
        training: { ...this.state.training, batchModeEnabled: false },
      });
      this.screen.render();
    });
    this.formElements.push(batchCheckbox);

    // Target Application Name
    const appLabel = blessed.text({
      parent: this.rightPanel,
      top: 10,
      left: 2,
      content: 'Target Application Name:',
    });
    this.formElements.push(appLabel);

    const appInputWrapper = blessed.box({
      parent: this.rightPanel,
      top: 11,
      left: 2,
      width: '90%',
      height: 3,
      border: { type: 'line' },
      style: { border: { fg: this.currentSelection === 2 ? 'white' : 'gray' } },
    });
    this.formElements.push(appInputWrapper);

    const appInput = blessed.textbox({
      parent: appInputWrapper,
      top: 0,
      left: 0,
      width: '100%-2',
      height: 1,
      inputOnFocus: true,
      style: { fg: 'white' },
      value: this.state.training.targetApplicationName,
    });
    appInput.on('submit', (value) => {
      this.onStateUpdate({
        training: { ...this.state.training, targetApplicationName: value },
      });
      this.isEditingField = false;
      if (this.updateFieldStyles) this.updateFieldStyles();
      this.screen.render();
    });
    appInput.on('cancel', () => {
      this.isEditingField = false;
      if (this.updateFieldStyles) this.updateFieldStyles();
      this.screen.render();
    });
    this.formElements.push(appInput);

    // Model Path
    const modelLabel = blessed.text({
      parent: this.rightPanel,
      top: 15,
      left: 2,
      content: 'Model Path:',
    });
    this.formElements.push(modelLabel);

    const modelInputWrapper = blessed.box({
      parent: this.rightPanel,
      top: 16,
      left: 2,
      width: '90%',
      height: 3,
      border: { type: 'line' },
      style: { border: { fg: this.currentSelection === 3 ? 'white' : 'gray' } },
    });
    this.formElements.push(modelInputWrapper);

    const modelInput = blessed.textbox({
      parent: modelInputWrapper,
      top: 0,
      left: 0,
      width: '100%-2',
      height: 1,
      inputOnFocus: true,
      style: { fg: 'white' },
      value: this.state.training.modelPath,
    });
    modelInput.on('submit', (value) => {
      this.onStateUpdate({
        training: { ...this.state.training, modelPath: value },
      });
      this.isEditingField = false;
      if (this.updateFieldStyles) this.updateFieldStyles();
      this.screen.render();
    });
    modelInput.on('cancel', () => {
      this.isEditingField = false;
      if (this.updateFieldStyles) this.updateFieldStyles();
      this.screen.render();
    });
    this.formElements.push(modelInput);

    // Training Controls Title
    const controlsLabel = blessed.text({
      parent: this.rightPanel,
      top: 20,
      left: 2,
      content: 'Training Controls:',
    });
    this.formElements.push(controlsLabel);

    // Start Training Button
    const startButton = blessed.button({
      parent: this.rightPanel,
      top: 21,
      left: 2,
      width: 15,
      height: 3,
      content: '{green-fg}Start Training{/}',
      tags: true,
      border: { type: 'line' },
      style: {
        border: { fg: this.currentSelection === 4 ? 'white' : 'gray' },
        focus: { border: { fg: 'cyan' } },
      },
    });
    startButton.on('press', () => {
      if (!this.state.training.isTrainingHappening) {
        this.onStateUpdate({
          training: { ...this.state.training, isTrainingHappening: true },
        });
        statusBox.setContent('{yellow-fg}Training in progress...{/}');
        this.screen.render();
      }
    });
    this.formElements.push(startButton);

    // Stop Training Button
    const stopButton = blessed.button({
      parent: this.rightPanel,
      top: 21,
      left: 19,
      width: 15,
      height: 3,
      content: '{red-fg}Stop Training{/}',
      tags: true,
      border: { type: 'line' },
      style: {
        border: { fg: this.currentSelection === 5 ? 'white' : 'gray' },
        focus: { border: { fg: 'cyan' } },
      },
    });
    stopButton.on('press', () => {
      if (this.state.training.isTrainingHappening) {
        this.onStateUpdate({
          training: { ...this.state.training, isTrainingHappening: false },
        });
        statusBox.setContent('{gray-fg}Training stopped{/}');
        this.screen.render();
      }
    });
    this.formElements.push(stopButton);

    // Training Status Area
    const statusLabel = blessed.text({
      parent: this.rightPanel,
      top: 26,
      left: 2,
      content: 'Training Status:',
    });
    this.formElements.push(statusLabel);

    const statusBox = blessed.box({
      parent: this.rightPanel,
      top: 27,
      left: 2,
      width: '90%',
      height: '40%',
      border: { type: 'line' },
      style: { border: { fg: 'gray' } },
      scrollable: true,
      alwaysScroll: true,
      mouse: true,
      content: this.state.training.isTrainingHappening ? 
        '{yellow-fg}Training in progress...{/}' : 
        '{gray-fg}Training not started{/}',
      tags: true,
    });
    this.formElements.push(statusBox);

    // Instructions
    const instructions = blessed.text({
      parent: this.rightPanel,
      bottom: 1,
      left: 2,
      content: '{gray-fg}Arrow keys to select • Enter to edit/activate • Space for checkbox • ESC for mode menu{/}',
      tags: true,
    });
    this.formElements.push(instructions);

    // Store references for navigation and activation
    const fieldWrappers = [dirInputWrapper, batchCheckboxWrapper, appInputWrapper, modelInputWrapper, startButton, stopButton];
    const focusableElements = [dirInput, batchCheckbox, appInput, modelInput, startButton, stopButton];
    
    // Initial highlighting
    this.updateFieldStyles = () => {
      fieldWrappers.forEach((wrapper, index) => {
        if (wrapper && wrapper.style) {
          if (index === 1) {
            // Checkbox wrapper
            wrapper.style.fg = (index === this.currentSelection) ? 'white' : 'gray';
          } else if (index === 4 || index === 5) {
            // Buttons
            wrapper.style.border = wrapper.style.border || {};
            wrapper.style.border.fg = (index === this.currentSelection) ? 'white' : 'gray';
          } else {
            // Text field wrappers
            wrapper.style.border = wrapper.style.border || {};
            wrapper.style.border.fg = (index === this.currentSelection && !this.isEditingField) ? 'white' : 'gray';
            if (index === this.currentSelection && this.isEditingField) {
              wrapper.style.border.fg = 'cyan';
            }
          }
        }
      });
      
      // Update checkbox text color
      if (batchCheckbox && batchCheckbox.style) {
        batchCheckbox.style.fg = (this.currentSelection === 1) ? 'white' : 'gray';
      }
      
      this.screen.render();
    };

    // Arrow key navigation within form
    const arrowHandler = (_ch: any, key: any) => {
      if (!this.isPanelFocused || this.isEditingField) return;
      
      if (key.name === 'up' && this.currentSelection > 0) {
        this.currentSelection--;
        if (this.updateFieldStyles) this.updateFieldStyles();
      } else if (key.name === 'down' && this.currentSelection < focusableElements.length - 1) {
        this.currentSelection++;
        if (this.updateFieldStyles) this.updateFieldStyles();
      }
    };

    // Enter key to activate field, Space only for checkbox
    const activateHandler = (_ch: any, key: any) => {
      if (!this.isPanelFocused || this.isEditingField) return;
      
      if (key.name === 'enter') {
        const element = focusableElements[this.currentSelection];
        if (element && element.parent) {
          if (this.currentSelection === 4) {
            // Handle start button
            startButton.press();
          } else if (this.currentSelection === 5) {
            // Handle stop button
            stopButton.press();
          } else if (this.currentSelection === 1) {
            // Handle checkbox
            batchCheckbox.toggle();
          } else {
            // Handle text fields
            this.isEditingField = true;
            if (this.updateFieldStyles) this.updateFieldStyles();
            element.focus();
          }
        }
      } else if (key.name === 'space' && this.currentSelection === 1) {
        // Space only works for checkbox
        batchCheckbox.toggle();
      }
    };

    this.screen.on('keypress', arrowHandler);
    this.screen.on('keypress', activateHandler);
    
    // Store handlers for cleanup
    this.tabHandler = () => {
      this.screen.off('keypress', arrowHandler);
      this.screen.off('keypress', activateHandler);
    };

    // Initial style update
    this.updateFieldStyles();
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
        '{gray-fg}Navigate with arrow keys • Press Enter to select{/}';

      return logo + welcome;
    }
    return '';
  }

  private getStatusText(): string {
    const currentModeLabel =
      this.modes.find((m) => m.value === this.state.currentMode)?.label || '';
    
    if (this.isPanelFocused) {
      return (
        ` Current Mode: ${currentModeLabel}\n` +
        ` Press ESC to return to mode selection | Use arrows/Tab to navigate form | Press 'q' to quit`
      );
    } else {
      return (
        ` Current Mode: ${currentModeLabel}\n` +
        ` Press 'q' to quit | Use arrow keys to select mode | Press Enter to choose`
      );
    }
  }

  private updateStatus(): void {
    this.statusBox.setContent(this.getStatusText());
    this.render();
  }

  private render(): void {
    this.screen.render();
  }

  public updateState(state: ApplicationState): void {
    const modeChanged = this.state.currentMode !== state.currentMode;
    this.state = state;
    this.updateStatus();
    
    // Only handle mode change if the mode actually changed
    if (modeChanged) {
      this.handleModeChange();
    }
  }

  public start(): void {
    this.handleModeChange();
  }
}