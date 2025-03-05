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

3. Run the main script:

```bash
./main.py  # or
python3 main.py
```

To enable persistent command history, preceed the command with `rlwrap` (you may need to install it first).

4. Print the help message:

```
[router-solver]# help
```

## Development Environment

1. Perform static type checking:

```bash
mypy **/*.py
```

2. Lint the code:

```bash
pylint **/*.py
```

3. Run all tests:

```bash
python3 -m unittest discover test
```

4. Run a specific test:

```bash
python3 -m unittest test.<test_module>
```