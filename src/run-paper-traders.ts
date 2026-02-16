#!/usr/bin/env node
/**
 * Launches multiple paper-trading modes side-by-side for faster label collection.
 *
 * Usage:
 *   npm run paper-trade-multi-both
 */

import { spawn } from 'node:child_process';
import path from 'node:path';

type Child = ReturnType<typeof spawn>;

const runChild = (label: string, scriptRelativeToCwd: string): Child => {
  const nodePath = process.execPath;
  const scriptPath = path.join(process.cwd(), scriptRelativeToCwd);

  const child = spawn(nodePath, [scriptPath], {
    stdio: 'inherit',
    env: {
      ...process.env,
      PAPER_TRADER_LABEL: label,
    },
  });

  return child;
};

const children: Child[] = [];

const killAll = (signal: NodeJS.Signals = 'SIGINT'): void => {
  for (const child of children) {
    if (!child.killed) {
      try {
        child.kill(signal);
      } catch {
        // ignore
      }
    }
  }
};

process.on('SIGINT', () => {
  killAll('SIGINT');
  process.exit(130);
});

process.on('SIGTERM', () => {
  killAll('SIGTERM');
  process.exit(143);
});

const swing = runChild('SWING', path.join('dist', 'multi-coin-paper-trader-swing.js'));
const scalp = runChild('SCALP', path.join('dist', 'multi-coin-paper-trader-scalp.js'));
children.push(swing, scalp);

let exited = false;
const onExit = (label: string) => (code: number | null, signal: NodeJS.Signals | null) => {
  if (exited) return;
  exited = true;

  console.log(`\n[paper-trade-multi-both] ${label} exited (code=${code ?? 'null'}, signal=${signal ?? 'null'}). Stopping others...`);
  killAll('SIGTERM');

  if (typeof code === 'number') process.exit(code);
  process.exit(1);
};

swing.on('exit', onExit('SWING'));
scalp.on('exit', onExit('SCALP'));
