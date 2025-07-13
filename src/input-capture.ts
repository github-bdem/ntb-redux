// import { createReadStream } from 'fs';
import { promisify } from 'util';
import { exec, spawn } from 'child_process';

const execAsync = promisify(exec);

export interface InputEvent {
  timestamp: number;
  type: 'keyboard' | 'mouse';
  action: 'press' | 'release' | 'move';
  key?: string;
  keyCode?: number;
  x?: number;
  y?: number;
  button?: 'left' | 'right' | 'middle';
}

export class InputCapture {
  private isCapturing = false;
  private events: InputEvent[] = [];
  private keyboardDevice: string | null = null;
  private mouseDevice: string | null = null;
  private processes: any[] = [];

  constructor() {}

  // Find input devices
  async findInputDevices(): Promise<void> {
    try {
      const { stdout } = await execAsync('cat /proc/bus/input/devices');
      const devices = this.parseInputDevices(stdout);

      // Find keyboard and mouse devices
      this.keyboardDevice = devices.find((d) => d.handlers.includes('kbd'))?.eventDevice || null;
      this.mouseDevice = devices.find((d) => d.handlers.includes('mouse'))?.eventDevice || null;

      console.log('Found devices:', {
        keyboard: this.keyboardDevice,
        mouse: this.mouseDevice,
      });
    } catch (error) {
      throw new Error(`Failed to find input devices: ${error}`);
    }
  }

  private parseInputDevices(
    deviceInfo: string,
  ): Array<{ name: string; handlers: string[]; eventDevice: string }> {
    const devices: Array<{ name: string; handlers: string[]; eventDevice: string }> = [];
    const sections = deviceInfo.split('\n\n');

    for (const section of sections) {
      const lines = section.split('\n');
      let name = '';
      let handlers: string[] = [];
      let eventDevice = '';

      for (const line of lines) {
        if (line.startsWith('N: Name=')) {
          name = line.substring(8).replace(/"/g, '');
        } else if (line.startsWith('H: Handlers=')) {
          handlers = line
            .substring(12)
            .split(' ')
            .filter((h) => h.length > 0);
        }
      }

      // Find event device from handlers
      const eventHandler = handlers.find((h) => h.startsWith('event'));
      if (eventHandler) {
        eventDevice = `/dev/input/${eventHandler}`;
        devices.push({ name, handlers, eventDevice });
      }
    }

    return devices;
  }

  // Start capturing input events
  async startCapturing(): Promise<void> {
    if (this.isCapturing) {
      throw new Error('Already capturing input events');
    }

    if (!this.keyboardDevice || !this.mouseDevice) {
      await this.findInputDevices();
    }

    this.isCapturing = true;
    this.events = [];

    // Start keyboard capture
    if (this.keyboardDevice) {
      this.captureKeyboardEvents();
    }

    // Start mouse capture
    if (this.mouseDevice) {
      this.captureMouseEvents();
    }

    console.log('Started input event capture');
  }

  private async captureKeyboardEvents(): Promise<void> {
    if (!this.keyboardDevice) return;

    try {
      // Use xinput to capture keyboard events (alternative to direct evdev)
      const { stdout } = await execAsync('xinput list --id-only "Virtual core keyboard"');
      const keyboardId = stdout.trim();

      const process = spawn('xinput', ['test', keyboardId]);
      this.processes.push(process);

      process.stdout?.on('data', (data) => {
        if (!this.isCapturing) return;

        const output = data.toString();
        const lines = output.split('\n');

        for (const line of lines) {
          if (line.includes('key press') || line.includes('key release')) {
            const parts = line.trim().split(/\s+/);
            const action = line.includes('press') ? 'press' : 'release';
            const keyCode = parseInt(parts[parts.length - 1]);

            this.events.push({
              timestamp: Date.now(),
              type: 'keyboard',
              action,
              keyCode,
              key: this.keyCodeToString(keyCode),
            });
          }
        }
      });

      process.stderr?.on('data', (data) => {
        console.error('xinput keyboard error:', data.toString());
      });
    } catch (error) {
      console.warn('Failed to capture keyboard events:', error);
    }
  }

  private async captureMouseEvents(): Promise<void> {
    if (!this.mouseDevice) return;

    try {
      // Use xinput to capture mouse events
      const { stdout } = await execAsync('xinput list --id-only "Virtual core pointer"');
      const mouseId = stdout.trim();

      const process = spawn('xinput', ['test', mouseId]);
      this.processes.push(process);

      process.stdout?.on('data', (data) => {
        if (!this.isCapturing) return;

        const output = data.toString();
        const lines = output.split('\n');

        for (const line of lines) {
          if (line.includes('button press') || line.includes('button release')) {
            const action = line.includes('press') ? 'press' : 'release';
            const buttonMatch = line.match(/button (\d+)/);
            const button = buttonMatch
              ? this.buttonNumberToString(parseInt(buttonMatch[1]))
              : undefined;

            this.events.push({
              timestamp: Date.now(),
              type: 'mouse',
              action,
              button,
            });
          } else if (line.includes('motion')) {
            // Parse mouse movement
            const coords = line.match(/(\d+\.\d+)\/(\d+\.\d+)/);
            if (coords) {
              this.events.push({
                timestamp: Date.now(),
                type: 'mouse',
                action: 'move',
                x: parseFloat(coords[1]),
                y: parseFloat(coords[2]),
              });
            }
          }
        }
      });

      process.stderr?.on('data', (data) => {
        console.error('xinput mouse error:', data.toString());
      });
    } catch (error) {
      console.warn('Failed to capture mouse events:', error);
    }
  }

  // Stop capturing and return collected events
  stopCapturing(): InputEvent[] {
    this.isCapturing = false;
    
    // Kill all processes
    this.processes.forEach((proc) => {
      if (!proc.killed) {
        proc.kill('SIGTERM');
      }
    });
    this.processes = [];
    
    console.log(`Captured ${this.events.length} input events`);
    return [...this.events];
  }

  // Get events since last call (streaming mode)
  getNewEvents(): InputEvent[] {
    const newEvents = [...this.events];
    this.events = [];
    return newEvents;
  }

  // Helper methods
  private keyCodeToString(keyCode: number): string {
    const keyMap: { [key: number]: string } = {
      9: 'Escape',
      10: '1',
      11: '2',
      12: '3',
      13: '4',
      14: '5',
      15: '6',
      16: '7',
      17: '8',
      18: '9',
      19: '0',
      24: 'q',
      25: 'w',
      26: 'e',
      27: 'r',
      28: 't',
      29: 'y',
      30: 'u',
      31: 'i',
      32: 'o',
      33: 'p',
      38: 'a',
      39: 's',
      40: 'd',
      41: 'f',
      42: 'g',
      43: 'h',
      44: 'j',
      45: 'k',
      46: 'l',
      52: 'z',
      53: 'x',
      54: 'c',
      55: 'v',
      56: 'b',
      57: 'n',
      58: 'm',
      65: 'space',
      36: 'Return',
      22: 'BackSpace',
      23: 'Tab',
      111: 'Up',
      116: 'Down',
      113: 'Left',
      114: 'Right',
    };
    return keyMap[keyCode] || `key_${keyCode}`;
  }

  private buttonNumberToString(buttonNum: number): 'left' | 'right' | 'middle' {
    switch (buttonNum) {
      case 1:
        return 'left';
      case 2:
        return 'middle';
      case 3:
        return 'right';
      default:
        return 'left';
    }
  }
}

// Alternative simpler approach using global key/mouse listeners

export class SimpleInputCapture {
  private events: InputEvent[] = [];
  private isCapturing = false;
  private processes: any[] = [];

  async startCapturing(): Promise<void> {
    this.isCapturing = true;
    this.events = [];

    // Capture keyboard using xev focused on root window
    const keyProcess = spawn('bash', ['-c', 'xev -root | grep -E "(KeyPress|KeyRelease)"']);
    keyProcess.stdout?.on('data', (data) => {
      if (!this.isCapturing) return;

      const output = data.toString();
      const lines = output.split('\n');

      for (const line of lines) {
        if (line.includes('KeyPress') || line.includes('KeyRelease')) {
          const action = line.includes('KeyPress') ? 'press' : 'release';
          const keyMatch = line.match(/keycode (\d+)/);
          const keyCode = keyMatch ? parseInt(keyMatch[1]) : 0;

          this.events.push({
            timestamp: Date.now(),
            type: 'keyboard',
            action,
            keyCode,
            key: this.keyCodeToString(keyCode),
          });
        }
      }
    });

    this.processes.push(keyProcess);
    console.log('Started simple input capture');
  }

  stopCapturing(): InputEvent[] {
    this.isCapturing = false;

    // Kill all processes
    this.processes.forEach((proc) => {
      if (!proc.killed) {
        proc.kill('SIGTERM');
      }
    });
    this.processes = [];

    console.log(`Captured ${this.events.length} input events`);
    return [...this.events];
  }

  private keyCodeToString(keyCode: number): string {
    // Same key mapping as above
    const keyMap: { [key: number]: string } = {
      9: 'Escape',
      10: '1',
      11: '2',
      12: '3',
      13: '4',
      14: '5',
      15: '6',
      16: '7',
      17: '8',
      18: '9',
      19: '0',
      24: 'q',
      25: 'w',
      26: 'e',
      27: 'r',
      28: 't',
      29: 'y',
      30: 'u',
      31: 'i',
      32: 'o',
      33: 'p',
      38: 'a',
      39: 's',
      40: 'd',
      41: 'f',
      42: 'g',
      43: 'h',
      44: 'j',
      45: 'k',
      46: 'l',
      52: 'z',
      53: 'x',
      54: 'c',
      55: 'v',
      56: 'b',
      57: 'n',
      58: 'm',
      65: 'space',
      36: 'Return',
      22: 'BackSpace',
      23: 'Tab',
      111: 'Up',
      116: 'Down',
      113: 'Left',
      114: 'Right',
    };
    return keyMap[keyCode] || `key_${keyCode}`;
  }
}
