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
./main.py  # or
python3 main.py
```

## Development Environment

1. Lint the code:

```bash
pylint src/**/*.py   # Source code
```

2. Run all tests:

```bash
python3 -m unittest discover test
```

3. Run a specific test:

```bash
python3 -m unittest test.<test_module>
```