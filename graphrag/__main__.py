"""python -m graphrag entry point"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from .pipeline import main
main()
