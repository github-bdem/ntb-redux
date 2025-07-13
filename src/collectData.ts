import { ScreenshotCapture } from './screenshot-capture.js';

class GameWindowCapture {
  private capture: ScreenshotCapture;
  private gameWindowId: string | null = null;

  constructor() {
    this.capture = new ScreenshotCapture();
  }

  async findGameWindow(titlePattern: string): Promise<void> {
    const windows = await this.capture.getWindows();
    const gameWindow = windows.find((w) =>
      w.title.toLowerCase().includes(titlePattern.toLowerCase()),
    );

    if (!gameWindow) {
      throw new Error(`Game window not found: ${titlePattern}`);
    }

    this.gameWindowId = gameWindow.id;
    console.log(`Found game window: ${gameWindow.title} (ID: ${gameWindow.id})`);
  }

  async captureGameFrame(): Promise<Buffer> {
    if (!this.gameWindowId) {
      throw new Error('Game window not found. Call findGameWindow() first.');
    }

    return await this.capture.captureWindowToBuffer(this.gameWindowId);
  }

  async startCapturing(
    intervalMs: number,
    onFrame: (imageBuffer: Buffer) => void,
  ): Promise<() => void> {
    if (!this.gameWindowId) {
      throw new Error('Game window not found. Call findGameWindow() first.');
    }

    const interval = setInterval(async () => {
      try {
        const frame = await this.captureGameFrame();
        onFrame(frame);
      } catch (error) {
        console.error('Frame capture error:', error);
      }
    }, intervalMs);

    // Return a way to stop capturing
    return () => clearInterval(interval);
  }
}

// Usage
const gameCapture = new GameWindowCapture();
await gameCapture.findGameWindow('nuclearthrone');

let numberOfPacketsToCapture = 10;
let i = 0;

const finish = await gameCapture.startCapturing(1000, (frame) => {
  // Process frame for ML inference
  console.log(`Captured frame: ${frame.length} bytes`);
  i += 1;
  if (i === numberOfPacketsToCapture) finish();
});
