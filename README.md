# feup-iart1
1st project for the IART course at FEUP

## Usage

1. Setup local Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```
   
2. Install the required packages:

```bash
pip install -r requirements.txt
```

1. Run the main script:

```bash
python3 src/app.py
```

## Development Environment

1. Lint the code:

```bash
pylint src/**/*.py   # Source code
pylint test/**/*.py  # Test code
```

2. Run the tests:

```bash
python3 -m unittest discover test
```