# MBPP Import Analysis: Pre-Execution Feasibility Study

**Date:** 2025-11-21
**Analyzed:** Phase 3.5 MBPP Temperature Robustness Results
**Total Code Samples:** 8,536 across 8 temperature variations (0.0 to 1.4)

---

## Executive Summary

### What Was Found
Analyzed all generated code in Phase 3.5 and discovered that **55 valid standard library imports** are used across the solutions. The model frequently uses libraries like `math` (49x), `re` (20x), `collections` (14x), and typing hints.

### The Problem
Currently, MBPP code execution does NOT pre-load any imports (unlike HumanEval which pre-loads from `data/phase0_3_humaneval/required_imports.json`). This means:
- Solutions using imports without explicit `import` statements will fail
- Evaluation results may be artificially lower due to import-related failures
- Inconsistent behavior between MBPP and HumanEval pipelines

### Key Decision Point
**Should we add pre-execution of standard library imports for MBPP?**

**Pros:**
- ✅ More fair evaluation (solutions won't fail due to missing imports)
- ✅ Consistent with HumanEval approach
- ✅ May improve pass@1 rates by reducing false negatives
- ✅ All imports are standard library (no dependency issues)

**Cons:**
- ❌ Requires re-running 6-10 phases (estimated 8-15 hours total)
- ❌ Results won't be comparable to existing runs
- ❌ May need to re-run statistical analysis phases (3.8, 4.14, etc.)

### Recommendation
**Consider impact vs. effort:**
1. **High Impact:** If current pass@1 rates are low and import errors are common → Re-run justified
2. **Low Impact:** If most solutions explicitly import what they need → May not be worth it
3. **Compromise:** Run Phase 1 on a small subset (--start 0 --end 50) with imports enabled, compare results

---

## Analysis Methodology

### Data Source
```bash
data/phase3_5/
├── dataset_temp_0_0.parquet (388 samples)
├── dataset_temp_0_2.parquet (1164 samples)
├── dataset_temp_0_4.parquet (1164 samples)
├── dataset_temp_0_6.parquet (1164 samples)
├── dataset_temp_0_8.parquet (1164 samples)
├── dataset_temp_1_0.parquet (1164 samples)
├── dataset_temp_1_2.parquet (1164 samples)
└── dataset_temp_1_4.parquet (1164 samples)
```

### Extraction Method
1. Read all `generated_code` columns from parquet files
2. Parse using Python AST (ast.parse) to extract Import/ImportFrom nodes
3. Fallback to regex for syntactically invalid code
4. Filter out hallucinated/invalid imports (found ~50 nonsensical ones)
5. Exclude test frameworks (unittest, pytest) and third-party libraries (numpy)

---

## Detailed Findings

### Top 20 Most Frequent Imports

| Count | Import Statement |
|-------|------------------|
| 49 | `import math` |
| 20 | `import re` |
| 15 | `import unittest` ⚠️ (excluded - test framework) |
| 14 | `import collections` |
| 14 | `from typing import List` |
| 11 | `import random` |
| 10 | `import sys` |
| 8 | `import numpy` ⚠️ (excluded - third-party) |
| 7 | `from collections import Counter` |
| 7 | `import itertools` |
| 7 | `import numpy as np` ⚠️ (excluded - third-party) |
| 6 | `import heapq` |
| 6 | `import pytest` ⚠️ (excluded - test framework) |
| 6 | `import string` |
| 6 | `from typing import Tuple` |
| 5 | `import functools` |
| 4 | `from collections import defaultdict` |
| 4 | `import operator` |
| 4 | `from functools import reduce` |
| 4 | `import time` |

### Complete List of Valid Standard Library Imports (55 total)

#### Module-Level Imports (15)
```python
import math           # 49x - Mathematical functions
import re             # 20x - Regular expressions
import collections    # 14x - Counter, defaultdict, deque, etc.
import random         # 11x - Random number generation
import sys            # 10x - System-specific parameters
import itertools      # 7x - Iterator functions
import functools      # 5x - Functional programming tools
import heapq          # 6x - Heap queue algorithm
import string         # 6x - String constants
import operator       # 4x - Operator functions
import time           # 4x - Time access
import copy           # 4x - Shallow/deep copy
import datetime       # 2x - Date and time types
import csv            # 2x - CSV file handling
import os             # 2x - OS interface
```

#### Specific Imports from collections (5)
```python
from collections import Counter        # 7x
from collections import defaultdict    # 4x
from collections import namedtuple     # 3x
from collections import deque          # 1x
from collections import OrderedDict    # 1x
```

#### Specific Imports from typing (5)
```python
from typing import List       # 14x - Most common type hint
from typing import Tuple      # 6x
from typing import Optional   # 2x
from typing import Callable   # 2x
from typing import Union      # 1x
```

#### Specific Imports from functools (2)
```python
from functools import reduce      # 4x
from functools import lru_cache   # 1x
```

#### Specific Imports from itertools (5)
```python
from itertools import combinations   # 3x
from itertools import permutations   # 1x
from itertools import chain          # 1x
from itertools import compress       # 1x
from itertools import filterfalse    # 1x
```

#### Specific Imports from math (5)
```python
from math import sqrt        # 3x
from math import ceil        # 2x
from math import factorial   # 2x
from math import gcd         # 1x
from math import hypot       # 1x
```

#### Specific Imports from bisect (3)
```python
from bisect import bisect_left   # 2x
from bisect import bisect        # 1x
from bisect import bisect_right  # 1x
```

#### Other Specific Imports (15)
```python
from copy import deepcopy              # 2x
from operator import itemgetter        # 1x
from operator import mul               # 1x
from string import ascii_lowercase     # 1x
from string import whitespace          # 1x
from re import findall                 # 1x
from re import match                   # 1x
from re import compile                 # 1x
from datetime import date              # 1x
from random import randint             # 1x
from random import randrange           # 1x
from sys import maxsize                # 1x
from heapq import heappush, heappop    # 1x
from __future__ import annotations     # 1x
from __future__ import print_function  # 1x
```

---

## Comparison with HumanEval

### HumanEval Import Pre-Loading
Location: `common_simplified/helpers.py:168-176`

```python
if config.dataset_name == "humaneval":
    import_file = Path("data/phase0_3_humaneval/required_imports.json")
    if import_file.exists():
        import json
        with open(import_file) as f:
            imports_data = json.load(f)
            import_code = '\n'.join(imports_data['imports'])
            exec(import_code, namespace)
```

### What HumanEval Imports (for reference)
Need to check: `data/phase0_3_humaneval/required_imports.json`

### MBPP vs HumanEval Import Patterns
**Similarities:**
- Both use `math`, `collections`, `itertools` frequently
- Both use type hints from `typing`
- Both use `functools.reduce`

**Differences:**
- MBPP uses `re` (regex) more heavily (20x vs likely 0x in HumanEval)
- MBPP uses `random` (11x) - may indicate randomized algorithms
- MBPP generated code occasionally imports `numpy` (third-party, excluded)

---

## Implementation Plan

### Step 1: Create Required Imports JSON
**File:** `data/phase0_3_mbpp/required_imports.json`

```json
{
  "imports": [
    "import math",
    "import re",
    "import collections",
    "import random",
    "import sys",
    "import itertools",
    "import functools",
    "import heapq",
    "import string",
    "import operator",
    "import time",
    "import copy",
    "import datetime",
    "import csv",
    "import os",
    "from collections import Counter",
    "from collections import defaultdict",
    "from collections import namedtuple",
    "from collections import deque",
    "from collections import OrderedDict",
    "from typing import List",
    "from typing import Tuple",
    "from typing import Optional",
    "from typing import Callable",
    "from typing import Union",
    "from functools import reduce",
    "from functools import lru_cache",
    "from itertools import combinations",
    "from itertools import permutations",
    "from itertools import chain",
    "from itertools import compress",
    "from itertools import filterfalse",
    "from math import sqrt",
    "from math import ceil",
    "from math import factorial",
    "from math import gcd",
    "from math import hypot",
    "from bisect import bisect_left",
    "from bisect import bisect",
    "from bisect import bisect_right",
    "from copy import deepcopy",
    "from operator import itemgetter",
    "from operator import mul",
    "from string import ascii_lowercase",
    "from string import whitespace",
    "from re import findall",
    "from re import match",
    "from re import compile",
    "from datetime import date",
    "from random import randint",
    "from random import randrange",
    "from sys import maxsize",
    "from heapq import heappush, heappop",
    "from __future__ import annotations",
    "from __future__ import print_function"
  ],
  "description": "Pre-scanned imports required for MBPP dataset execution",
  "source": "Phase 3.5 generated code analysis (8536 samples)",
  "total_codes_analyzed": 8536,
  "unique_imports": 55,
  "analysis_date": "2025-11-21"
}
```

### Step 2: Update Code Evaluation Helper
**File:** `common_simplified/helpers.py` (around line 168)

**Current Code:**
```python
if config.dataset_name == "humaneval":
    import_file = Path("data/phase0_3_humaneval/required_imports.json")
    if import_file.exists():
        import json
        with open(import_file) as f:
            imports_data = json.load(f)
            import_code = '\n'.join(imports_data['imports'])
            exec(import_code, namespace)
```

**Proposed Addition:**
```python
if config.dataset_name == "humaneval":
    import_file = Path("data/phase0_3_humaneval/required_imports.json")
    if import_file.exists():
        import json
        with open(import_file) as f:
            imports_data = json.load(f)
            import_code = '\n'.join(imports_data['imports'])
            exec(import_code, namespace)

elif config.dataset_name == "mbpp":
    import_file = Path("data/phase0_3_mbpp/required_imports.json")
    if import_file.exists():
        import json
        with open(import_file) as f:
            imports_data = json.load(f)
            import_code = '\n'.join(imports_data['imports'])
            exec(import_code, namespace)
```

---

## Pipeline Re-Run Cost Analysis

### Phases That Would Need Re-Running

#### Category 1: Data Generation (MUST re-run)
These phases generate the core dataset with evaluations:

| Phase | Name | Estimated Time | Why Re-run? |
|-------|------|----------------|-------------|
| 1 | Baseline Generation | 2-6 hours | New import pre-loading changes evaluation results |
| 3.5 | Temperature Robustness | 3-8 hours | Different pass@1 rates at each temperature |

**Subtotal: 5-14 hours**

#### Category 2: Statistical Analysis (MUST re-run)
These phases depend on the pass@1 classifications:

| Phase | Name | Estimated Time | Why Re-run? |
|-------|------|----------------|-------------|
| 3.8 | AUROC/F1 Evaluation | 10-30 min | Feature rankings may change |
| 3.10 | Temperature AUROC | 10-30 min | Per-temperature metrics |
| 3.12 | Difficulty AUROC | 10-30 min | Stratified analysis |

**Subtotal: 30-90 minutes**

#### Category 3: Causal Validation (MUST re-run)
These phases depend on feature rankings from Phase 3.8:

| Phase | Name | Estimated Time | Why Re-run? |
|-------|------|----------------|-------------|
| 4.5 | Steering Coefficients | 1-2 hours | May select different coefficients |
| 4.6 | Golden Section Refinement | 30-60 min | Refines coefficients |
| 4.8 | Steering Effect Analysis | 2-4 hours | Tests steering with new baseline |
| 4.10 | Random Feature Control | 1-2 hours | Control experiment |
| 4.12 | Random Feature Steering | 1-2 hours | Control steering |
| 4.14 | Statistical Significance | 15-30 min | Binomial tests |

**Subtotal: 6-11 hours**

#### Category 4: Optional (May Skip)
These phases might not be critically affected:

| Phase | Name | Impact |
|-------|------|--------|
| 2.2 | Pile Activation Cache | No impact (separate dataset) |
| 2.5 | SAE Analysis | No impact (no pass@1 dependency) |
| 2.10 | T-Statistic Selection | No impact (uses Phase 1 activations only) |
| 5.x | Weight Orthogonalization | Possibly affected (depends on feature rankings) |
| 6.3 | Attention Patterns | Possibly affected |
| 7.x | Instruction-Tuned Baseline | Would need re-run for IT model |

---

### Total Estimated Re-Run Time

**Minimum (Core Pipeline):**
- Phase 1: 2 hours (if only 487 problems)
- Phase 3.5: 3 hours
- Phase 3.8: 15 minutes
- Phase 4.8: 2 hours
- **Total: ~7-8 hours**

**Full Statistical Pipeline:**
- All phases 1, 3.5, 3.8, 3.10, 3.12, 4.5-4.14
- **Total: ~12-16 hours**

**With Instruction-Tuned Model:**
- Add phases 7.3-7.12
- **Total: ~18-25 hours**

---

## Risk Assessment

### Low Risk Changes
✅ All imports are Python standard library (no dependency issues)
✅ Import pre-execution is already proven with HumanEval
✅ Code changes are minimal (1 JSON file + 10 lines of code)

### Medium Risk Factors
⚠️ Results won't be comparable to existing baseline
⚠️ May discover that impact is minimal (wasted re-run time)
⚠️ Re-running may introduce new variability

### High Risk Factors
❌ Re-running requires ~8-16 hours of compute time
❌ May need to update paper/results if already drafted
❌ Could cascade to requiring Phase 0.1 re-split if dataset changes

---

## Recommended Next Steps

### Option 1: Quick Validation Test (Recommended)
**Time:** 30-60 minutes
**Risk:** Low

1. Implement the import changes (JSON + helpers.py)
2. Run Phase 1 on small subset: `python3 run.py phase 1 --start 0 --end 50`
3. Compare pass@1 rates with/without imports
4. If improvement > 5%, proceed with full re-run
5. If improvement < 2%, skip re-run

### Option 2: Full Re-Run Immediately
**Time:** 12-16 hours
**Risk:** Medium

- Justified if: Current pass@1 rates are suspiciously low
- Justified if: You observe many "NameError: 'math' not defined" style failures
- Justified if: Paper/results not yet finalized

### Option 3: Defer Until Next Experiment
**Time:** 0 hours now
**Risk:** Low

- Apply changes for future experiments (LLAMA, HumanEval, etc.)
- Keep current MBPP results as-is for consistency
- Document the limitation in paper/appendix

---

## Code Change Summary

### Files to Create
1. `data/phase0_3_mbpp/required_imports.json` (55 import statements)

### Files to Modify
1. `common_simplified/helpers.py` (add 10 lines around line 168)

### No Changes Needed To
- `common/config.py` (no new config parameters)
- `run.py` (no new CLI arguments)
- Any phase runners (changes are in shared utility)

---

## Appendix A: Examples of Potentially Affected Code

### Example 1: Math Usage Without Import
```python
def calculate_distance(x1, y1, x2, y2):
    return sqrt((x2-x1)**2 + (y2-y1)**2)
```
**Current behavior:** NameError: name 'sqrt' is not defined
**With imports:** Works (if `from math import sqrt` is pre-loaded)

### Example 2: Regex Usage Without Import
```python
def extract_numbers(text):
    return findall(r'\d+', text)
```
**Current behavior:** NameError: name 'findall' is not defined
**With imports:** Works (if `from re import findall` is pre-loaded)

### Example 3: Collections Usage
```python
def count_frequency(items):
    return Counter(items)
```
**Current behavior:** NameError: name 'Counter' is not defined
**With imports:** Works (if `from collections import Counter` is pre-loaded)

---

## Appendix B: Raw Analysis Output

```
Analyzing 8 files...

✓ dataset_temp_0_0.parquet: 388 code samples
✓ dataset_temp_0_2.parquet: 1164 code samples
✓ dataset_temp_0_4.parquet: 1164 code samples
✓ dataset_temp_0_6.parquet: 1164 code samples
✓ dataset_temp_0_8.parquet: 1164 code samples
✓ dataset_temp_1_0.parquet: 1164 code samples
✓ dataset_temp_1_2.parquet: 1164 code samples
✓ dataset_temp_1_4.parquet: 1164 code samples

Total codes analyzed: 8536
Total import statements: 359
Unique imports: 147 (after filtering → 55 valid)
```

### Filtering Breakdown
- **359** total import statements found
- **147** unique import statements
- **~50** hallucinated/nonsensical (e.g., "from smallest 7 and for 6...")
- **~20** test framework imports (unittest, pytest, testdata)
- **~15** third-party libraries (numpy, scipy, pygraphviz)
- **~7** custom/non-existent modules (regex, utils, myutils)
- **55** valid standard library imports ✅

---

## Appendix C: Comparison with Current Codebase

### Where Import Pre-Loading Is Currently Used

**File:** `common_simplified/helpers.py:168-176`

```python
if config.dataset_name == "humaneval":
    import_file = Path("data/phase0_3_humaneval/required_imports.json")
    if import_file.exists():
        import json
        with open(import_file) as f:
            imports_data = json.load(f)
            import_code = '\n'.join(imports_data['imports'])
            # Execute imports in namespace
            exec(import_code, namespace)
```

**Observation:** MBPP is not handled, so no imports are pre-loaded for MBPP evaluations.

---

## Document Metadata

- **Created:** 2025-11-21
- **Author:** Claude Code Analysis
- **Analysis Script:** Ad-hoc Python (AST + regex parsing)
- **Data Source:** `data/phase3_5/*.parquet` (8 files)
- **Sample Size:** 8,536 code solutions
- **Temperature Range:** 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4

---

## Questions for Reflection

1. **What is the current pass@1 rate for Phase 1 MBPP?**
   - If it's already high (>70%), impact may be minimal
   - If it's low (<50%), imports could make a significant difference

2. **Have you observed NameError exceptions in Phase 1/3.5 logs?**
   - Check for "NameError: name 'math' is not defined" patterns
   - If many such errors exist, imports will help significantly

3. **How critical is result consistency with prior runs?**
   - If results are already published/shared: May want to defer
   - If still in active development: Good time to add this

4. **What is the timeline for the research?**
   - If deadline is tight: Skip re-run, document as limitation
   - If time permits: Run validation test first (Option 1)

5. **Are you planning to run experiments with other datasets/models soon?**
   - If yes (LLAMA, other datasets): Apply changes now for future runs
   - If no: May be worth doing the re-run for thoroughness
