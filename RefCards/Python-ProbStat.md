# Commonly Used Python Modules for Probability & Statistics

## Sets

+ [Set definition](../Stats/ProbStatsPython/02-Sets.md#21-notation)
  + define a set: `{...}` or `set(...)`
    + e.g., `Set1 = {1, 2}; print(Set1) $ {1, 2}`, `Set2 = set({2, 3}); print(Set2) ${2, 3}`
  + empty set: using only `set()` or `set({})`
    + e.g., `Empty1 = set(); type(Empty1) # set; print(Empty1) # set{}`
    + e.g., `Empty2 = set({}); type(Empty2) # set; print(Empty2) # set{}`
    + e.g., `NotASet = {}; type(NotASet) # dict`, `{}` not an empty set

+ [Membership](../Stats/ProbStatsPython/02-Sets.md#21-notation)
  + $\in$: `in`
  + $\notin$: `not in`

+ [Testing if empty set, size](../Stats/ProbStatsPython/02-Sets.md#21-notation)
  + test empty: `not`
  + size: `len()`
  + check if size is 0: `len() == 0`

+ [Intervals and Multiples](../Stats/ProbStatsPython/02-Sets.md#22-basic-sets)
  + $\{0, \dots, n-1\}$: `range(n)`
  + $\{m, \dots, n-1\}$: `range(m, n)`
  + $\{m,\, m+d,\, m+2d,\, \dots\} \leq n-1$: `range(m, n, d)`

+ [Set relationship](../Stats/ProbStatsPython/02-Sets.md#24-relations)
  + equality ($= \;\to$ `==`)
  + inequality ($\neq \;\to$ `!=`)
  + disjoint (`isdisjoint`)

+ [Subsets and supersets](../Stats/ProbStatsPython/02-Sets.md#24-relations)
  + subset ($\subseteq\; \to$ `<=` or `issubset`)
  + supserset ($\supseteq\; \to$ `>=` or `issuperset`)
  + strict subset ($\subset\; \to$ `<`)
  + strict supeerset ($\supset\; \to$ `>`)

+ [Union and Intersection](../Stats/ProbStatsPython/02-Sets.md#25-operations)
  + union ($\cup \to$ `|` or `union`)
  + intersection ($\cap \to$ `&` or `intersection`)

+ [Set- and Symmetric-Difference](../Stats/ProbStatsPython/02-Sets.md#25-operations)
  + set difference ($- \to$ `-` or `difference`)
  + symmetric difference ($\Delta \to$ `^` or `symmetric_difference`)

+ [Set operations - Summary](../Stats/ProbStatsPython/02-Sets.md#25-operations)
  + complement: $A^c$
  + intersection: $\cap \to$ `&` or `intersection`
  + union: $\cup \to$ `|` or `union`
  + difference: $- \to$ `-` or `difference`
  + symmetric difference: $\Delta \to$ `^` or `symmetric_difference`

+ [Cartesian products](../Stats/ProbStatsPython/02-Sets.md#26-cartesian-products)
  + tuples: $(a_1, \dots, a_n)$
  + ordered pairs: $(a, b)$
  + Python: `product` functon in `itertools` library



