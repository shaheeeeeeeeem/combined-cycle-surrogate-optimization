# Surrogate-Assisted Optimization of a Methane-Fired Combined Cycle

ME3160 Power & Refrigeration Cycles, IIT Hyderabad.

Physics-based model of a methane-fired gas turbine (Brayton topping cycle) with a heat recovery
steam generator and a single-pressure Rankine bottoming cycle, followed by design of experiments,
surrogate modelling and genetic-algorithm optimization of the overall efficiency.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Tests

```powershell
pytest
```
