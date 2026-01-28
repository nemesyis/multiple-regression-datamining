This multiple linear regression calculator was created to facilitate the analysis of relationships between variables using the regression method:
- Raka Farza Pratama
- Rifqi Adam Safarudin
- Hammam Arrizal

University of Serang Raya (UNSERA) - Indonesia

Built with Streamlit
You do not need a database for this program.
## Prerequisites

Before running this application, you need:

- **Python 3.7 or higher** (Python 3.9+ recommended)
- **Anaconda** (recommended) or pip package manager
- Basic knowledge of CSV file formats

## Installation

### Option 1: Using Anaconda (Recommended)

1. **Install Anaconda**
   - Download from: https://www.anaconda.com/download
   - Follow the installation wizard for your operating system

2. **Create a Virtual Environment**
```bash
   conda create -n regression_app python=3.11
```

3. **Activate the Environment**
```bash
   conda activate regression_app
```

4. **Install Required Packages**
```bash
   conda install numpy pandas matplotlib scikit-learn
   pip install streamlit
```

### Option 2: Using pip (Alternative)

1. **Create a Virtual Environment**
```bash
   # On Windows
   python -m venv regression_env
   regression_env\Scripts\activate

   # On Mac/Linux
   python3 -m venv regression_env
   source regression_env/bin/activate
```

2. **Install Required Packages**
```bash
   pip install -r requirements.txt
```

## Usage

### Starting the Application

1. **Navigate to Project Directory**
```bash
   cd path/to/your/project
```

2. **Activate Your Environment**
```bash
   # Anaconda
   conda activate regression_app

   # pip virtual environment (Windows)
   regression_env\Scripts\activate

   # pip virtual environment (Mac/Linux)
   source regression_env/bin/activate
```

3. **Run the Application**
```bash
   streamlit run regression_app.py
```

4. **Access the Application**
   - The app will automatically open in your default browser
   - If not, navigate to: `http://localhost:8501`
