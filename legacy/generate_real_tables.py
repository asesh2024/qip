import numpy as np
import os

encoders = ['basis', 'angle', 'entangling'] 
# Using exact float values as they format in Python strings (0.0, 0.05, 0.1, 0.2)
epsilons = [0.0, 0.05, 0.1, 0.2]
tau = 0.65
target_states = ['00000', '01110', '10110', '11021']

table_s2_data = {}

print("\n" + "="*80)
print("MAIN TABLE (TABLE 7) - TRUE LATEX ROWS")
print("="*80)

for enc in encoders:
    for eps in epsilons:
        # Load the arrays you just saved
        try:
            X_test = np.load(f"X_te_{enc}_{eps}.npy")
            y_prob = np.load(f"probs_{enc}_{eps}.npy")
        except FileNotFoundError:
            print(f"  % Missing data for {enc} {eps}")
            continue
        
        # Enforce L1 simplex normalization
        y_prob = y_prob / np.sum(y_prob, axis=1, keepdims=True)
        max_p = np.max(y_prob, axis=1)
        
        # Exact statistics
        mean_p = np.mean(max_p)
        std_p = np.std(max_p)
        median_p = np.median(max_p)
        min_p = np.min(max_p)
        max_p_val = np.max(max_p)
        
        # Margins
        if min_p >= tau:
            margin = f"+{(min_p - tau):.3f}"
        else:
            margin = f"+{(tau - max_p_val):.3f}"
            
        triggered = np.sum(max_p < tau)
        trigger_rate = (triggered / len(max_p)) * 100
        
        # Display epsilon with two decimals for LaTeX (e.g., 0.00, 0.10)
        print(f"  & $\\epsilon = {eps:.2f}$ & ${mean_p:.3f} \\pm {std_p:.3f}$ & {median_p:.3f} & $[{min_p:.3f}, {max_p_val:.3f}]$ & {margin} & {triggered:,} & {trigger_rate:.1f}\\% \\\\")

        # Process discrete states for Table S2
        str_features = ["".join(map(lambda x: str(int(x)), row)) for row in X_test]
        unique_states, indices, counts = np.unique(str_features, return_index=True, return_counts=True)
        
        state_dict = {}
        for state, idx, count in zip(unique_states, indices, counts):
            freq = (count / len(max_p)) * 100
            state_dict[state] = (max_p[idx], freq)
            
        table_s2_data[(enc, eps)] = state_dict
        
    print("  \\midrule")

print("\n" + "="*80)
print("SUPPLEMENTARY TABLE S2 - TRUE LATEX ROWS")
print("="*80)

for enc in encoders:
    for eps in epsilons:
        if (enc, eps) not in table_s2_data:
            continue
            
        state_dict = table_s2_data[(enc, eps)]
        row_strs = []
        gate_status = "Triggered" if np.max([v[0] for v in state_dict.values()]) < tau else "Bypassed"
        
        for state in target_states:
            if state in state_dict:
                val, freq = state_dict[state]
                row_strs.append(f"{val:.3f} ({freq:.1f}\\%)")
            else:
                row_strs.append("N/A (0.0\\%)")
                
        latex_cols = " & ".join(row_strs)
        print(f"  & $\\epsilon = {eps:.2f}$ & {latex_cols} & {gate_status} \\\\")
        
    print("  \\midrule")
