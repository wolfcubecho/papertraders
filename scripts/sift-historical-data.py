#!/usr/bin/env python3
"""
Historical Data Sifting Script
Scans the historical data archive and copies only files for target symbols and timeframes.
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Set, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data-sifting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Target symbols to filter
TARGET_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
    'TONUSDT', 'ADAUSDT', 'TRXUSDT', 'AVAXUSDT', 'LINKUSDT'
]

# Target timeframes
TARGET_TIMEFRAMES = ['1m', '5m', '1h', '1d']

class DataSifter:
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.stats = {
            'files_scanned': 0,
            'files_copied': 0,
            'bytes_processed': 0,
            'bytes_copied': 0,
            'skipped_files': []
        }
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Source directory: {self.source_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Target symbols: {', '.join(TARGET_SYMBOLS)}")
        logger.info(f"Target timeframes: {', '.join(TARGET_TIMEFRAMES)}")

    def should_include_file(self, filepath: Path) -> Tuple[bool, str]:
        """
        Check if a file should be included based on symbol and timeframe.
        Returns (should_include, reason)
        """
        filename = filepath.name.lower()
        
        # Check if filename contains any target symbol
        matched_symbol = None
        for symbol in TARGET_SYMBOLS:
            if symbol.lower() in filename:
                matched_symbol = symbol
                break
        
        if not matched_symbol:
            return False, "No matching symbol"
        
        # Check if filename contains target timeframe
        matched_timeframe = None
        for tf in TARGET_TIMEFRAMES:
            if tf in filename:
                matched_timeframe = tf
                break
        
        if not matched_timeframe:
            return False, "No matching timeframe"
        
        return True, f"{matched_symbol} - {matched_timeframe}"

    def process_directory(self, dir_path: Path) -> None:
        """
        Recursively process a directory and copy matching files.
        """
        try:
            for item in dir_path.iterdir():
                if item.is_file():
                    self.process_file(item)
                elif item.is_dir():
                    self.process_directory(item)
        except PermissionError as e:
            logger.warning(f"Permission denied: {dir_path} - {e}")
        except Exception as e:
            logger.error(f"Error processing directory {dir_path}: {e}")

    def process_file(self, filepath: Path) -> None:
        """
        Process a single file: check if it matches criteria and copy if so.
        """
        self.stats['files_scanned'] += 1
        file_size = filepath.stat().st_size
        self.stats['bytes_processed'] += file_size
        
        # Show progress every 100 files
        if self.stats['files_scanned'] % 100 == 0:
            logger.info(f"Scanned {self.stats['files_scanned']} files, copied {self.stats['files_copied']} files")
        
        # Check if file should be included
        should_include, reason = self.should_include_file(filepath)
        
        if not should_include:
            self.stats['skipped_files'].append(f"{filepath.name}: {reason}")
            return
        
        # Determine output path
        # Extract symbol and timeframe from filename
        matched_symbol = None
        for symbol in TARGET_SYMBOLS:
            if symbol.lower() in filepath.name.lower():
                matched_symbol = symbol
                break
        
        matched_timeframe = None
        for tf in TARGET_TIMEFRAMES:
            if tf in filepath.name:
                matched_timeframe = tf
                break
        
        # Create output path: filtered/SYMBOL/TIMEFRAME/filename
        output_path = self.output_dir / matched_symbol / matched_timeframe / filepath.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        try:
            shutil.copy2(filepath, output_path)
            self.stats['files_copied'] += 1
            self.stats['bytes_copied'] += file_size
            logger.debug(f"Copied: {filepath.name} -> {output_path}")
        except Exception as e:
            logger.error(f"Failed to copy {filepath.name}: {e}")

    def run(self) -> None:
        """
        Run the data sifting process.
        """
        logger.info("=" * 60)
        logger.info("Starting data sifting process")
        logger.info("=" * 60)
        
        # Check if source directory exists
        if not self.source_dir.exists():
            logger.error(f"Source directory does not exist: {self.source_dir}")
            return
        
        # Process source directory (will handle all subdirectories recursively)
        logger.info(f"Processing source directory: {self.source_dir}")
        self.process_directory(self.source_dir)
        
        # Print summary
        self.print_summary()
        
        # Save skipped files log
        if self.stats['skipped_files']:
            with open(self.output_dir / 'skipped-files.log', 'w') as f:
                f.write('\n'.join(self.stats['skipped_files']))
            logger.info(f"Saved {len(self.stats['skipped_files'])} skipped file entries to skipped-files.log")

    def print_summary(self) -> None:
        """
        Print a summary of the sifting process.
        """
        logger.info("=" * 60)
        logger.info("Data Sifting Complete - Summary")
        logger.info("=" * 60)
        logger.info(f"Files scanned:     {self.stats['files_scanned']:,}")
        logger.info(f"Files copied:      {self.stats['files_copied']:,}")
        logger.info(f"Bytes processed:    {self.format_bytes(self.stats['bytes_processed'])}")
        logger.info(f"Bytes copied:       {self.format_bytes(self.stats['bytes_copied'])}")
        
        if self.stats['bytes_processed'] > 0:
            logger.info(f"Reduction ratio:    {(1 - self.stats['bytes_copied'] / self.stats['bytes_processed'] * 100):.1f}%")
        else:
            logger.info(f"Reduction ratio:    N/A (no files processed)")
        
        logger.info(f"Output directory:   {self.output_dir}")
        logger.info("=" * 60)
        
        # List what we've organized
        logger.info("\nOrganized data structure:")
        for symbol in TARGET_SYMBOLS:
            symbol_dir = self.output_dir / symbol
            if symbol_dir.exists():
                logger.info(f"  {symbol}/")
                for tf in TARGET_TIMEFRAMES:
                    tf_dir = symbol_dir / tf
                    if tf_dir.exists():
                        file_count = len(list(tf_dir.glob('*')))
                        logger.info(f"    {tf}/ - {file_count} files")

    @staticmethod
    def format_bytes(bytes_count: int) -> str:
        """
        Format bytes into human-readable format.
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_count < 1024.0:
                return f"{bytes_count:.2f} {unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.2f} PB"


def main():
    parser = argparse.ArgumentParser(
        description='Sift historical data for target symbols and timeframes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sift-historical-data.py
  python sift-historical-data.py --source "C:\\path\\to\\Historical Data" --output "./filtered"
        """
    )
    
    parser.add_argument(
        '--source',
        default='C:/Users/caleb/trading-stack/mcp/learning-orchestrator/Historical Data',
        help='Source directory containing historical data'
    )
    
    parser.add_argument(
        '--output',
        default='./Historical Data/filtered',
        help='Output directory for filtered data (default: ./Historical Data/filtered)'
    )
    
    args = parser.parse_args()
    
    # Run sifter
    sifter = DataSifter(args.source, args.output)
    sifter.run()


if __name__ == '__main__':
    main()