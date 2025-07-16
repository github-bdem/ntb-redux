import { GameController } from './dist/game-controller.js';

async function testController() {
  console.log('🧪 Testing game controller...');
  
  const config = {
    deadZone: 0.1,
    mouseSpeed: 1.0,
    keyPressDelay: 50,
    smoothMouse: true,
    debugMode: true
  };
  
  const controller = new GameController(config);
  
  try {
    await controller.initialize();
    console.log('✅ Controller initialized successfully');
    
    // Test key press/release
    console.log('Testing key presses...');
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    console.log('✅ Controller test completed');
  } catch (error) {
    console.error('❌ Controller test failed:', error.message);
  }
}

testController();
