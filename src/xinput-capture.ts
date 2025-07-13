import { spawn, ChildProcess } from 'child_process';
import { InputEvent } from './input-capture.js';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export class XInputCapture {
  private events: InputEvent[] = [];
  private isCapturing = false;
  private processes: ChildProcess[] = [];
  private keyboardDevices: string[] = [];

  async startCapturing(): Promise<void> {
    this.isCapturing = true;
    this.events = [];

    try {
      // Find keyboard devices
      await this.findKeyboardDevices();

      // Start capturing from each keyboard device
      for (const deviceId of this.keyboardDevices) {
        this.captureFromDevice(deviceId);
      }

      console.log(`Started xinput capture with ${this.keyboardDevices.length} keyboard device(s)`);
    } catch (error) {
      console.error('Failed to start xinput capture:', error);
    }
  }

  private async findKeyboardDevices(): Promise<void> {
    try {
      const { stdout } = await execAsync('xinput list --short');
      const lines = stdout.split('\n');
      
      for (const line of lines) {
        // Look for keyboard devices
        if (line.includes('keyboard') && !line.includes('Virtual') && line.includes('id=')) {
          const idMatch = line.match(/id=(\d+)/);
          if (idMatch && idMatch[1]) {
            this.keyboardDevices.push(idMatch[1]);
            console.log(`Found keyboard device: ${line.trim()}`);
          }
        }
      }
    } catch (error) {
      console.error('Failed to list xinput devices:', error);
    }
  }

  private captureFromDevice(deviceId: string): void {
    try {
      const process = spawn('xinput', ['test', deviceId]);
      
      process.stdout?.on('data', (data) => {
        if (!this.isCapturing) return;
        this.parseXInputOutput(data.toString());
      });

      process.stderr?.on('data', (data) => {
        console.error(`xinput error for device ${deviceId}:`, data.toString().trim());
      });

      process.on('error', (error) => {
        console.error(`Failed to spawn xinput for device ${deviceId}:`, error.message);
      });

      this.processes.push(process);
    } catch (error) {
      console.error(`Failed to capture from device ${deviceId}:`, error);
    }
  }

  private parseXInputOutput(output: string): void {
    const lines = output.split('\n');
    
    for (const line of lines) {
      // Parse xinput test output
      // key press   17 
      // key release 17
      
      if (line.includes('key press') || line.includes('key release')) {
        const action = line.includes('press') ? 'press' : 'release';
        const keyCodeMatch = line.match(/key (?:press|release)\s+(\d+)/);
        
        if (keyCodeMatch && keyCodeMatch[1]) {
          const keyCode = parseInt(keyCodeMatch[1]);
          const keyName = this.keyCodeToName(keyCode);
          
          this.events.push({
            timestamp: Date.now(),
            type: 'keyboard',
            action,
            keyCode,
            key: keyName
          });
          
          console.log(`Key event: ${keyName} ${action} (code: ${keyCode})`);
        }
      }
    }
  }

  private keyCodeToName(keyCode: number): string {
    // X11 keycodes to key names mapping
    const keyMap: { [key: number]: string } = {
      // Letters
      24: 'q', 25: 'w', 26: 'e', 27: 'r', 28: 't', 29: 'y', 30: 'u', 31: 'i', 32: 'o', 33: 'p',
      38: 'a', 39: 's', 40: 'd', 41: 'f', 42: 'g', 43: 'h', 44: 'j', 45: 'k', 46: 'l',
      52: 'z', 53: 'x', 54: 'c', 55: 'v', 56: 'b', 57: 'n', 58: 'm',
      
      // Numbers
      10: '1', 11: '2', 12: '3', 13: '4', 14: '5', 15: '6', 16: '7', 17: '8', 18: '9', 19: '0',
      
      // Special keys
      9: 'escape',
      22: 'backspace',
      23: 'tab',
      36: 'enter',
      50: 'shift_l',
      62: 'shift_r',
      37: 'ctrl_l',
      105: 'ctrl_r',
      64: 'alt_l',
      108: 'alt_r',
      65: 'space',
      
      // Arrow keys
      111: 'up',
      116: 'down',
      113: 'left',
      114: 'right',
      
      // Function keys
      67: 'f1', 68: 'f2', 69: 'f3', 70: 'f4', 71: 'f5', 72: 'f6',
      73: 'f7', 74: 'f8', 75: 'f9', 76: 'f10', 95: 'f11', 96: 'f12',
    };
    
    return keyMap[keyCode] || `key_${keyCode}`;
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

  getNewEvents(): InputEvent[] {
    const newEvents = [...this.events];
    this.events = [];
    return newEvents;
  }
}