# Gingivitis Project  
A neural‑network powered tool employing U‑Net++ architecture to analyse and detect gingivitis disease.

## Description  
This project implements a segmented‑deep‑learning pipeline using U‑Net++ to identify indications of gingivitis in medical imaging. It is written in Python and offers an easy‑to‑use interface (`main_screen.py`) for launching the application and downloading all required dependencies.

## Installation & Setup  

### Part 1 – Installing Python Interpreter (Skip this part if already installed)  
1. Visit [python.org/downloads](https://www.python.org/downloads/) to download the latest stable Python 3 release. ([python.org](https://www.python.org/downloads/?utm_source=chatgpt.com))  
2. You should use **Python 3.13.x** (for example, Python 3.13.7) as your interpreter version. ([python.org](https://www.python.org/downloads/release/python-3137/?utm_source=chatgpt.com))  
3. Install Python and ensure that the `python3` (or `python`) command is available in your terminal or command‑prompt.  
4. Verify your installation by running:  
   ```bash  
   python3 --version  
   ```  
   The output should show a version like `Python 3.13.x`.

### Part 2 – Running the Application  
1. Navigate to the `UI` directory (where `main_screen.py` is located).  
2. On your first run, execute:  
   ```bash  
   python3 main_screen.py
   ```  
   This initial run will download all required dependencies (e.g., Python packages, models).  
3. After the dependencies are installed, subsequent runs are identical. Simply execute the same command:  
   ```bash  
   python3 main_screen.py
   ```  
   The application should then start smoothly without further setup steps.

## Usage Example  
```bash  
python3 main_screen.py
```

## Authors  
- Amir Sorani  
- Daniel Yehoshua  
