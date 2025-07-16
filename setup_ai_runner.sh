#!/bin/bash

echo "🎮 Setting up Nuclear Throne AI Runner..."

# Check if xdotool is installed
if ! command -v xdotool &> /dev/null; then
    echo "📦 Installing xdotool..."
    sudo apt update
    sudo apt install -y xdotool
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install xdotool"
        exit 1
    fi
    
    echo "✅ xdotool installed successfully"
else
    echo "✅ xdotool already installed"
fi

# Test xdotool functionality
echo "🧪 Testing xdotool functionality..."
xdotool version > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ xdotool working correctly"
else
    echo "❌ xdotool test failed"
    exit 1
fi

# Building latest
echo "Building latest ntb-r version"
npm i
npm run build

# Create test script for controller
echo "📝 Creating controller test script..."
cat > test-controller.ts << 'EOF'
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
EOF

# Create AI testing script
echo "📝 Creating AI test script..."
cat > test-ai.ts << 'EOF'
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
EOF

echo "✅ Setup completed!"
echo ""
echo "🚀 Next steps:"
echo ""
echo "1. Test the controller (without a model):"
echo "   npx tsx test-controller.ts"
echo ""
echo "2. Test the AI system (requires trained model):"
echo "   npx tsx test-ai.ts"
echo ""
echo "3. Run the full AI (make sure Nuclear Throne is running):"
echo "   npx tsx nuclear-throne-ai.ts ./model --debug"
echo ""
echo "4. Run AI without controller (safe testing):"
echo "   npx tsx nuclear-throne-ai.ts ./model --no-controller --debug"
echo ""
echo "⚠️  Important notes:"
echo "- Make sure Nuclear Throne is running and visible"
echo "- Update the model path in test-ai.ts to your actual model location"
echo "- Start with --no-controller flag for safe testing"
echo "- Use Ctrl+C to stop the AI at any time"
