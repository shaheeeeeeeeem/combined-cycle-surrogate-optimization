# textbook_tables — reference property data (READ-ONLY)

Property tables from **Borgnakke & Sonntag, *Fundamentals of Thermodynamics*, 7th Ed. (B&S), Appendix A**, and a worked combined-cycle example from **Çengel & Boles, *Thermodynamics: An Engineering Approach***.

- The `.png` files are screenshots of the textbook pages and are the ground truth. They are kept locally and not committed.
- The `.csv` files are hand transcriptions of those screenshots, for code to load.
- If a CSV value ever disagrees with its PNG, the PNG is correct.

## Files

| File | What it is | Used for |
|---|---|---|
| `BS7e_TableA6_cp_polynomials.png` / `.csv` | Table A.6: ideal-gas cp polynomials | Variable-cp properties of every gas species (Phase 1 onward) |
| `BS7e_TableA7.1_air_ideal_gas.png` / `.csv` | Table A.7.1: ideal-gas properties of air (T, u, h, s°_T) | Check-1 hand values, reference-state shift, spot checks only |
| `Cengel_Ex10-8_combined_cycle_p1.png`, `_p2.png` / `Cengel_Ex10-8_combined_cycle.csv` | Çengel & Boles, Example 10-8 "A Combined Gas–Steam Power Cycle": inputs and published results | Textbook combined-cycle validation (Phase 4) |

## Table A.6: how to use it

```
Cp0 = C0 + C1·θ + C2·θ² + C3·θ³    [kJ/kg·K]    θ = T[K] / 1000
```

- Coefficients are on a **mass basis**, kJ/kg·K.
- **Validity: 250 K – 1200 K** (table footnote). Refrigerant rows are valid only up to 500 K.
- This project goes above 1200 K: TIT reaches 1500 K, and Check-1 uses T₃ = 1400 K. Above 1200 K the polynomial is **extrapolated**. The report must state this.
- Species this project needs: **N₂, O₂, CO₂, H₂O (the "Steam" row)**. CH₄ is listed but not needed, because the fuel enters at 298 K.
- There is also an **"Air" row**. The project spec defines air as 21/79 O₂/N₂ by mole, so air is modelled as that mixture built from the N₂ and O₂ rows. The Air row is used only for side-by-side comparison.

Integrated forms, with θ = T/1000 and T in K:

```
h(T)  = 1000·[C0·θ + C1·θ²/2 + C2·θ³/3 + C3·θ⁴/4]          (+ const)   [kJ/kg]
s°(T) = C0·ln T + C1·θ + C2·θ²/2 + C3·θ³/3                  (+ const)   [kJ/kg·K]
```

Shift both so that h = s° = 0 at 298.15 K (the project reference state).

## Table A.7.1: notes

- Reference state is B&S's own: h = 298.62 kJ/kg and s°_T = 6.86305 kJ/kg·K at 298.15 K.
- Our code uses 298.15 K = 0. To compare absolute values, subtract the 298.15 K row.
- The transcription was checked by testing h − u ≈ R·T (R = 0.287 kJ/kg·K) on all 77 rows. Deviations were ≤ 0.09 kJ/kg and grew smoothly with T, which suggests no typos.

## Çengel Example 10-8: notes

- The gas side is **air-standard** (no combustion), using Çengel's own air table, so expect ~0.5–1% differences from the A.6 21/79 mixture in gas-side h-based quantities.
- The steam side is an **ideal** Rankine cycle (η_pump = η_turbine = 1).
- The HRSG exit gas temperature is **fixed at 450 K** (no pinch closure). Steam flow follows from y = (h4 − h5)/(h3 − h2).
- η_cc is defined here as w_net / q_in (the gas heat input), not on a fuel-LHV basis.
