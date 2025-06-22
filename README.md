# Artificial Intelligence Project 1 - Router Placement

> Curricular Unit: [Artificial Intelligence](https://sigarra.up.pt/feup/en/UCURR_GERAL.FICHA_UC_VIEW?pv_ocorrencia_id=541894)<br>
> Faculty: [FEUP](https://sigarra.up.pt/feup/en/web_page.Inicial)<br>
> Professor: [Pedro Mota](https://sigarra.up.pt/feup/en/func_geral.formview?p_codigo=671784)<br>
> Authors: [Bruno Oliveira](https://github.com/Process-ing), [Henrique Fernandes](https://github.com/HenriqueSFernandes), [Rodrigo Silva](https://github.com/racoelhosilva)<br>
> Final Grade: 20.0/20<br>

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

## Tips and Tricks (for anyone doing a similar project)

- If you haven't already pick a theme, please really think before choosing an optimization problem. These have objectively harder algorithms than the solitary/adversarial games, and the base (the non-AI part) of the project is already quite tedious, so really think about it before choosing.
- Even for optimization problems, using `pygame` can be really useful for visualization, and we really recommend it over other recommended frameworks like Qt. If you need to make charts (e.g. for plotting the score over time), `pygame-chart` is actually really good for that: since it is built on top of `pygame`, it integrates really well and seems to also be quite performant.
- Most of the optimization problems have well documented solutions on papers online, which is great if you are unsure of your approach. Also, don't worry if your implementation is not very good, the professors only care that the algorithms are correct, following the respective meta-heuristics.
