#!/usr/bin/env python
"""GraphRAG entry point. Usage: python main.py or python -m graphrag"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

from graphrag.pipeline import main
main()
