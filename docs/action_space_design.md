# Action Space Design for ICU Offline RL

## Recommended Approach: 3-Phase Implementation

### **PHASE 1: Minimal Action Space (START HERE)** ✅

**Why:** Simplest for initial offline DQN, easier to debug, faster training

**Action Space:** 4 discrete actions focused on **vasopressor management**

```
Action 0: No vasopressor / Maintain current
Action 1: Low-dose vasopressor (e.g., norepinephrine < 0.1 mcg/kg/min)
Action 2: Medium-dose vasopressor (0.1-0.3 mcg/kg/min)
Action 3: High-dose vasopressor (> 0.3 mcg/kg/min)
```

**Rationale:**
- Vasopressors are the most critical intervention for hemodynamic instability
- Clear clinical meaning (dose escalation)
- ~97% of your ICU episodes have treatment data
- Easy to extract from INPUTEVENTS_CV/MV

**Implementation:**
```python
def discretize_vasopressor_action(rate_mcg_kg_min):
    """Map continuous rate to discrete action."""
    if pd.isna(rate_mcg_kg_min) or rate_mcg_kg_min == 0:
        return 0  # No vasopressor
    elif rate_mcg_kg_min < 0.1:
        return 1  # Low dose
    elif rate_mcg_kg_min < 0.3:
        return 2  # Medium dose
    else:
        return 3  # High dose
```

**Pros:**
- Simple, interpretable
- Matches clinical practice (dose titration)
- Sufficient for proof-of-concept
- Low computational cost

**Cons:**
- Ignores other treatments (fluids, sedation)
- May be too coarse for nuanced policies

---

### **PHASE 2: Dual-Dimension Action Space (EXPAND TO THIS)**

**Why:** Adds fluid management (second most important ICU intervention)

**Action Space:** 12 discrete actions (3 vasopressor × 4 fluid)

**Dimension 1 - Vasopressor:**
```
0: None
1: Low dose
2: High dose
```

**Dimension 2 - IV Fluid:**
```
0: Maintenance only (<100 mL/hr)
1: Moderate fluids (100-250 mL/hr)
2: Fluid bolus (250-500 mL/hr)
3: Aggressive resuscitation (>500 mL/hr)
```

**Combined Actions:** 3 × 4 = **12 total actions**

**Example Mapping:**
```python
def encode_action(vaso_level, fluid_level):
    """Encode 2D action into single discrete action."""
    return vaso_level * 4 + fluid_level

def decode_action(action_id):
    """Decode discrete action back to components."""
    vaso_level = action_id // 4
    fluid_level = action_id % 4
    return vaso_level, fluid_level
```

**Pros:**
- Captures two key interventions
- Still manageable action space size
- More realistic clinical decision-making

**Cons:**
- 3× more actions than Phase 1
- Requires extracting both vasopressor and fluid rates
- Potential action sparsity (not all combinations used)

---

### **PHASE 3: Full Clinical Action Space (FUTURE WORK)**

**Why:** Most realistic, but complex for offline RL

**Action Space:** Multi-dimensional or hierarchical

**Option 3A: Multi-Dimensional Discrete (48 actions)**
```
Dimension 1 - Vasopressor: [None, Low, Medium, High] (4 levels)
Dimension 2 - Fluid: [Maintenance, Moderate, Bolus] (3 levels)
Dimension 3 - Sedation: [None, Light, Moderate, Deep] (4 levels)

Total: 4 × 3 × 4 = 48 discrete actions
```

**Option 3B: Continuous Action Space**
```
Action = [vasopressor_rate, fluid_rate, sedation_dose]
         ∈ [0, 1]³  (normalized continuous values)

Requires: TD3, SAC, or CQL with continuous actions
```

**Pros:**
- Most comprehensive
- Can learn nuanced policies
- Closer to actual clinical practice

**Cons:**
- Large action space → sample efficiency issues
- Continuous actions harder for offline RL (behavioral cloning baseline needed)
- Requires more data and compute

---

## **My Specific Recommendation for You**

### **Start with Phase 1 (4 Vasopressor Actions)**

Here's why this is the best starting point:

1. **Focus on Core Problem**: Sepsis/shock management is primarily about vasopressor titration
2. **Data Availability**: You have good vasopressor data in INPUTEVENTS
3. **Clinical Validity**: Matches how clinicians think ("Should I escalate pressors?")
4. **Offline RL Friendly**: 4 actions → less behavior policy mismatch
5. **Debugging**: Easy to interpret what the agent learns
6. **Fast Iteration**: Train DQN in <1 hour, quickly test reward functions

### **Implementation Roadmap**

**Week 1-2: Phase 1 - Vasopressor Only**
```python
# src/actions/vasopressor_discretizer.py

VASOPRESSOR_ITEMIDS = {
    30047, 30120,  # Norepinephrine (CareVue)
    221906,        # Norepinephrine (MetaVision)
    # Add epinephrine, dopamine, etc.
}

def extract_vasopressor_action(inputevents, hour_start, hour_end):
    """Extract ground-truth action for a time window."""
    hour_events = inputevents[
        (inputevents['charttime'] >= hour_start) &
        (inputevents['charttime'] < hour_end) &
        (inputevents['itemid'].isin(VASOPRESSOR_ITEMIDS))
    ]

    if len(hour_events) == 0:
        return 0  # No vasopressor

    # Use mean rate during the hour
    mean_rate = hour_events['rate'].mean()

    # Discretize
    if mean_rate < 0.1:
        return 1
    elif mean_rate < 0.3:
        return 2
    else:
        return 3
```

**Week 3-4: Validate & Analyze**
- Check action distribution (is it balanced?)
- Compute behavior policy (what % of each action?)
- Analyze action sequences (do they make clinical sense?)
- Correlation with outcomes

**Week 5+: Phase 2 - Add Fluids**
- Expand to 12 actions
- Retrain and compare performance

---

## **Expected Action Distribution (Phase 1)**

Based on typical ICU data, you'll likely see:

```
Action 0 (No vasopressor):     ~40-50% of hours
Action 1 (Low dose):           ~20-30%
Action 2 (Medium dose):        ~15-20%
Action 3 (High dose):          ~5-10%
```

**This is good for offline RL:**
- Not too sparse (all actions represented)
- Not too uniform (there's a behavior policy to learn)
- Matches clinical conservatism (start low, escalate if needed)

---

## **Validation Strategy**

Before training RL, validate your action extraction:

```python
# Sanity checks
def validate_action_extraction(episodes_df):
    """Check if extracted actions make sense."""

    # 1. Action distribution
    action_counts = episodes_df['action'].value_counts()
    print("Action distribution:")
    print(action_counts / len(episodes_df))

    # 2. Action transitions (should be mostly ±1)
    transitions = episodes_df.groupby('icustay_id')['action'].apply(
        lambda x: (x.diff().abs() <= 1).mean()
    )
    print(f"Smooth transitions: {transitions.mean():.1%}")

    # 3. Correlation with outcome
    mortality_by_action = episodes_df.groupby('action')['died'].mean()
    print("Mortality by max action used:")
    print(mortality_by_action)

    # 4. Temporal pattern (escalation over time?)
    early_actions = episodes_df[episodes_df['hour'] < 6]['action'].mean()
    late_actions = episodes_df[episodes_df['hour'] >= 6]['action'].mean()
    print(f"Early hours avg action: {early_actions:.2f}")
    print(f"Late hours avg action: {late_actions:.2f}")
```

---

## **Summary Table**

| Phase | Actions | Complexity | When to Use |
|-------|---------|------------|-------------|
| **1: Vasopressor Only** | 4 | Low | Initial DQN baseline, proof-of-concept |
| **2: Vaso + Fluid** | 12 | Medium | After Phase 1 works, for better policies |
| **3: Full Multi-Dim** | 48+ | High | Research goal, after Phase 2 validated |

---

## **Final Recommendation**

**Start here:**
```
Action Space = {0: No Pressor, 1: Low, 2: Medium, 3: High}
Focus = Norepinephrine dosing
Goal = Predict survival with learned policy
```

**Success criteria before moving to Phase 2:**
1. Action extraction works (no bugs, makes clinical sense)
2. Offline DQN trains without errors
3. Learned policy beats random baseline
4. Learned policy comparable to behavior cloning
5. Action choices interpretable (e.g., escalates when BP drops)

Then expand to Phase 2 for better performance.

---

## **Code Structure**

```
src/actions/
  ├── __init__.py
  ├── discretizer.py          # Phase 1: Vasopressor discretization
  ├── extractor.py            # Extract ground-truth actions from data
  ├── validator.py            # Sanity checks
  └── multidim.py             # Phase 2+: Multi-dimensional actions
```

Would you like me to implement Phase 1 action extraction code now?
