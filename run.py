#!/usr/bin/env python
"""GraphRAG entry point. Usage: python run.py"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

from graphrag.pipeline import main
main()
