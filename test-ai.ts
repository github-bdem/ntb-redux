import { NuclearThroneAI } from './dist/nuclear-throne-ai.js';

async function testAI() {
  console.log('🧪 Testing Nuclear Throne AI (dry run)...');
  
  const config = {
    modelPath: './model', // Update this path
    gameWindowTitle: 'nuclearthrone',
    targetFPS: 10,
    enableController: false, // Disable for testing
    safetyMode: true,
    performance: {
      smoothingFactor: 0.3,
      confidenceThreshold: 0.4,
      deadZone: 0.1,
      mouseSpeed: 0.8
    },
    debug: {
      enabled: true,
      logActions: true,
      saveSession: false
    }
  };
  
  const ai = new NuclearThroneAI(config);
  
  try {
    await ai.initialize();
    console.log('✅ AI initialized successfully');
    
    // Run for 5 seconds as test
    console.log('Running AI test for 5 seconds...');
    setTimeout(async () => {
      await ai.stop();
      console.log('✅ AI test completed');
    }, 5000);
    
    await ai.start();
    
  } catch (error) {
    console.error('❌ AI test failed:', error.message);
  }
}

testAI();
