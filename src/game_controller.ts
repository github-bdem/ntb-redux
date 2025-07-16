#!/usr/bin/env ts-node

import { spawn } from 'child_process';
import { GameAction } from './realtime-inference.js';

interface ControllerConfig {
  deadZone: number; // Minimum movement to register
  mouseSpeed: number; // Mouse movement speed multiplier
  keyPressDelay: number; // Delay between key presses (ms)
  smoothMouse: boolean; // Enable mouse movement smoothing
  debugMode: boolean;
}

interface KeyState {
  w: boolean;
  a: boolean;
  s: boolean;
  d: boolean;
  shooting: boolean;
}

class GameController {
  private config: ControllerConfig;
  private currentKeyState: KeyState = {
    w: false,
    a: false,
    s: false,
    d: false,
    shooting: false
  };
  private lastMousePosition = { x: 0, y: 0 };
  private isEnabled = true;

  constructor(config: ControllerConfig) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    console.log('🎮 Initializing game controller...');
    
    // Test if we can execute input commands
    try {
      await this.testInputSystem();
      console.log('✅ Game controller ready');
    } catch (error) {
      throw new Error(`Failed to initialize input system: ${error.message}`);
    }
  }

  private async testInputSystem(): Promise<void> {
    // Test xdotool availability
    try {
      await this.executeCommand('xdotool', ['version']);
    } catch (error) {
      throw new Error('xdotool not found. Install with: sudo apt install xdotool');
    }
  }

  async executeAction(action: GameAction, gameWindowId: string): Promise<void> {
    if (!this.isEnabled) return;

    try {
      // Focus on game window first
      await this.focusGameWindow(gameWindowId);

      // Execute movement
      await this.handleMovement(action.movement);

      // Execute aiming
      await this.handleAiming(action.aim, gameWindowId);

      // Execute shooting
      await this.handleShooting(action.shooting);

      if (this.config.debugMode) {
        this.logControllerState(action);
      }

    } catch (error) {
      console.error('❌ Controller error:', error.message);
    }
  }

  private async focusGameWindow(windowId: string): Promise<void> {
    // Focus the game window to ensure inputs go to the right place
    await this.executeCommand('xdotool', ['windowfocus', windowId]);
  }

  private async handleMovement(movement: { x: number; y: number }): Promise<void> {
    // Apply dead zone
    const magnitude = Math.sqrt(movement.x * movement.x + movement.y * movement.y);
    if (magnitude < this.config.deadZone) {
      // Release all movement keys if below dead zone
      await this.releaseAllMovementKeys();
      return;
    }

    // Determine which keys should be pressed
    const newKeyState = {
      w: movement.y < -this.config.deadZone,  // Up
      s: movement.y > this.config.deadZone,   // Down
      a: movement.x < -this.config.deadZone,  // Left
      d: movement.x > this.config.deadZone    // Right
    };

    // Press/release keys as needed
    await this.updateMovementKeys(newKeyState);
  }

  private async updateMovementKeys(newState: Partial<KeyState>): Promise<void> {
    const keys = ['w', 'a', 's', 'd'] as const;
    
    for (const key of keys) {
      const shouldPress = newState[key] || false;
      const currentlyPressed = this.currentKeyState[key];
      
      if (shouldPress && !currentlyPressed) {
        // Press key
        await this.pressKey(key);
        this.currentKeyState[key] = true;
      } else if (!shouldPress && currentlyPressed) {
        // Release key
        await this.releaseKey(key);
        this.currentKeyState[key] = false;
      }
    }
  }

  private async releaseAllMovementKeys(): Promise<void> {
    const keys = ['w', 'a', 's', 'd'] as const;
    
    for (const key of keys) {
      if (this.currentKeyState[key]) {
        await this.releaseKey(key);
        this.currentKeyState[key] = false;
      }
    }
  }

  private async handleAiming(aim: { x: number; y: number }, gameWindowId: string): Promise<void> {
    // Calculate target mouse position relative to game window
    const targetX = Math.max(0, Math.min(320, aim.x));
    const targetY = Math.max(0, Math.min(240, aim.y));

    if (this.config.smoothMouse) {
      // Smooth mouse movement
      const deltaX = (targetX - this.lastMousePosition.x) * this.config.mouseSpeed;
      const deltaY = (targetY - this.lastMousePosition.y) * this.config.mouseSpeed;
      
      if (Math.abs(deltaX) > 1 || Math.abs(deltaY) > 1) {
        await this.moveMouseRelative(deltaX, deltaY);
        this.lastMousePosition.x += deltaX;
        this.lastMousePosition.y += deltaY;
      }
    } else {
      // Direct mouse positioning
      await this.moveMouseToPosition(targetX, targetY, gameWindowId);
      this.lastMousePosition = { x: targetX, y: targetY };
    }
  }

  private async handleShooting(shouldShoot: boolean): Promise<void> {
    if (shouldShoot && !this.currentKeyState.shooting) {
      // Start shooting
      await this.pressMouseButton(1); // Left mouse button
      this.currentKeyState.shooting = true;
    } else if (!shouldShoot && this.currentKeyState.shooting) {
      // Stop shooting
      await this.releaseMouseButton(1);
      this.currentKeyState.shooting = false;
    }
  }

  private async pressKey(key: string): Promise<void> {
    await this.executeCommand('xdotool', ['keydown', key]);
    
    if (this.config.debugMode) {
      console.log(`🔽 Key pressed: ${key}`);
    }
  }

  private async releaseKey(key: string): Promise<void> {
    await this.executeCommand('xdotool', ['keyup', key]);
    
    if (this.config.debugMode) {
      console.log(`🔼 Key released: ${key}`);
    }
  }

  private async pressMouseButton(button: number): Promise<void> {
    await this.executeCommand('xdotool', ['mousedown', button.toString()]);
    
    if (this.config.debugMode) {
      console.log(`🖱️ Mouse button pressed: ${button}`);
    }
  }

  private async releaseMouseButton(button: number): Promise<void> {
    await this.executeCommand('xdotool', ['mouseup', button.toString()]);
    
    if (this.config.debugMode) {
      console.log(`🖱️ Mouse button released: ${button}`);
    }
  }

  private async moveMouseToPosition(x: number, y: number, windowId: string): Promise<void> {
    // Move mouse to absolute position within the window
    await this.executeCommand('xdotool', [
      'mousemove',
      '--window', windowId,
      x.toString(),
      y.toString()
    ]);
  }

  private async moveMouseRelative(deltaX: number, deltaY: number): Promise<void> {
    // Move mouse relative to current position
    await this.executeCommand('xdotool', [
      'mousemove_relative',
      Math.round(deltaX).toString(),
      Math.round(deltaY).toString()
    ]);
  }

  private async executeCommand(command: string, args: string[]): Promise<string> {
    return new Promise((resolve, reject) => {
      const process = spawn(command, args);
      let stdout = '';
      let stderr = '';

      process.stdout?.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr?.on('data', (data) => {
        stderr += data.toString();
      });

      process.on('close', (code) => {
        if (code === 0) {
          resolve(stdout.trim());
        } else {
          reject(new Error(`Command failed: ${command} ${args.join(' ')}\nError: ${stderr}`));
        }
      });

      process.on('error', (error) => {
        reject(new Error(`Failed to execute command: ${error.message}`));
      });
    });
  }

  private logControllerState(action: GameAction): void {
    const movementKeys = [];
    if (this.currentKeyState.w) movementKeys.push('W');
    if (this.currentKeyState.a) movementKeys.push('A');
    if (this.currentKeyState.s) movementKeys.push('S');
    if (this.currentKeyState.d) movementKeys.push('D');
    
    console.log(`🎮 Keys: [${movementKeys.join(',')}] Mouse: (${this.lastMousePosition.x.toFixed(0)},${this.lastMousePosition.y.toFixed(0)}) Shoot: ${this.currentKeyState.shooting}`);
  }

  // Emergency stop - release all inputs
  async emergencyStop(): Promise<void> {
    console.log('🚨 Emergency stop - releasing all inputs');
    
    await this.releaseAllMovementKeys();
    
    if (this.currentKeyState.shooting) {
      await this.releaseMouseButton(1);
      this.currentKeyState.shooting = false;
    }
  }

  // Enable/disable controller
  setEnabled(enabled: boolean): void {
    this.isEnabled = enabled;
    
    if (!enabled) {
      this.emergencyStop();
    }
    
    console.log(`🎮 Controller ${enabled ? 'enabled' : 'disabled'}`);
  }

  // Get current input state for debugging
  getCurrentState(): KeyState & { mousePosition: typeof this.lastMousePosition; enabled: boolean } {
    return {
      ...this.currentKeyState,
      mousePosition: { ...this.lastMousePosition },
      enabled: this.isEnabled
    };
  }
}

// Utility class for safe controller management
class SafeGameController extends GameController {
  private safetyTimer: NodeJS.Timeout | null = null;
  private lastActionTime = Date.now();
  private readonly MAX_INACTIVE_TIME = 5000; // 5 seconds

  constructor(config: ControllerConfig) {
    super(config);
    this.startSafetyMonitor();
  }

  private startSafetyMonitor(): void {
    this.safetyTimer = setInterval(() => {
      const timeSinceLastAction = Date.now() - this.lastActionTime;
      
      if (timeSinceLastAction > this.MAX_INACTIVE_TIME) {
        console.log('⚠️  No actions received recently, safety stop');
        this.emergencyStop();
      }
    }, 1000);
  }

  async executeAction(action: GameAction, gameWindowId: string): Promise<void> {
    this.lastActionTime = Date.now();
    await super.executeAction(action, gameWindowId);
  }

  destroy(): void {
    if (this.safetyTimer) {
      clearInterval(this.safetyTimer);
      this.safetyTimer = null;
    }
    
    this.emergencyStop();
  }
}

export { GameController, SafeGameController, ControllerConfig, KeyState };