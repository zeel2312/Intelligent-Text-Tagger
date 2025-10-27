"""
Run the Streamlit web interface
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""
    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "web/app.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ])

if __name__ == "__main__":
    main()