#!/usr/bin/env node
/**
 * Learning Orchestrator CLI
 * Simple commands for the ML trading system
 */

import { H2OIntegration } from './h2o-integration.js';
import fs from 'fs';
import path from 'path';

async function main() {
  const command = process.argv[2];
  const args = process.argv.slice(3);

  switch (command) {
    case 'status':
      await showStatus();
      break;

    case 'h2o-train':
      await trainH2O(args);
      break;

    case 'h2o-predict':
      await predictH2O(args);
      break;

    case 'h2o-list':
      await listModels();
      break;

    case 'help':
    default:
      showHelp();
      break;
  }
}

async function showStatus() {
  const modelDir = path.join(process.cwd(), 'data', 'models');
  const trainingDir = path.join(process.cwd(), 'data', 'h2o-training');
  const loopDir = path.join(process.cwd(), 'data', 'learning-loop');

  console.log('\n=== Learning Orchestrator Status ===\n');

  // Check best model
  const bestModelPath = path.join(modelDir, 'best-model.json');
  if (fs.existsSync(bestModelPath)) {
    const model = JSON.parse(fs.readFileSync(bestModelPath, 'utf-8'));
    console.log('Best Model:');
    console.log(`  ID: ${model.modelId || 'unknown'}`);
    console.log(`  Accuracy: ${model.accuracy ? (model.accuracy * 100).toFixed(1) + '%' : model.finalAccuracy ? (model.finalAccuracy * 100).toFixed(1) + '%' : 'N/A'}`);
    console.log(`  Trained: ${model.trainedAt ? new Date(model.trainedAt).toISOString() : model.timestamp || 'N/A'}`);
  } else {
    console.log('Best Model: None trained yet');
    console.log('  Run: npm run learn-loop');
  }

  // Check training data
  console.log('\nTraining Data:');
  if (fs.existsSync(trainingDir)) {
    const files = fs.readdirSync(trainingDir).filter(f => f.endsWith('.csv'));
    console.log(`  CSV files: ${files.length}`);
  } else {
    console.log('  No training data directory');
  }

  // Check loop history
  console.log('\nLearning Loop:');
  if (fs.existsSync(loopDir)) {
    const files = fs.readdirSync(loopDir).filter(f => f.startsWith('loop_history'));
    if (files.length > 0) {
      const latest = files.sort().reverse()[0];
      const history = JSON.parse(fs.readFileSync(path.join(loopDir, latest), 'utf-8'));
      console.log(`  Last run: ${history.timestamp}`);
      console.log(`  Iterations: ${history.iterations?.length || 0}`);
      console.log(`  Trades used: ${history.totalTrades || 0}`);
    } else {
      console.log('  No loop history');
    }
  } else {
    console.log('  Not run yet');
  }

  console.log('\n=== Commands ===');
  console.log('  npm run learn-loop     # Run self-improving backtest loop');
  console.log('  npm run h2o-pipeline   # Extract + train (one-time)');
  console.log('  npm run h2o-service    # Start background learning');
  console.log('  npm run ml-advisor     # Test ML predictions');
}

async function trainH2O(args: string[]) {
  const csvPath = args[0];
  const targetColumn = args[1] || 'outcome';
  const algorithm = args[2] || 'gbm';

  if (!csvPath) {
    console.log('Usage: cli h2o-train <csv_path> [target_column] [algorithm]');
    return;
  }

  const h2o = new H2OIntegration();
  const result = await h2o.trainModel(csvPath, targetColumn, algorithm as any);
  console.log('Training result:', JSON.stringify(result, null, 2));
}

async function predictH2O(args: string[]) {
  const modelId = args[0];
  const featuresJson = args[1];

  if (!modelId || !featuresJson) {
    console.log('Usage: cli h2o-predict <model_id> <features_json>');
    return;
  }

  const h2o = new H2OIntegration();
  const features = JSON.parse(featuresJson);
  const result = await h2o.predict(modelId, features);
  console.log('Prediction:', JSON.stringify(result, null, 2));
}

async function listModels() {
  const h2o = new H2OIntegration();
  const models = await h2o.listModels();
  console.log('Models:', JSON.stringify(models, null, 2));
}

function showHelp() {
  console.log(`
Learning Orchestrator CLI

Commands:
  status        Show system status
  h2o-train     Train an H2O model
  h2o-predict   Make a prediction
  h2o-list      List available models
  help          Show this help

NPM Scripts:
  npm run learn-loop     Self-improving backtest loop
  npm run h2o-pipeline   One-time historical extraction + training
  npm run h2o-service    Background continuous learning
  npm run ml-advisor     Test ML predictions
`);
}

main().catch(console.error);
