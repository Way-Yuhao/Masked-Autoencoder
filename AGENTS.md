## Python Formatting

- Prefer single-line function signatures and function calls when they fit within 99 columns.
- Keep compact signatures and calls compact. Do not split arguments one-per-line unless line length or readability requires it.
- If multiline formatting is needed, group multiple arguments per line when practical instead of using one argument per line.
- Keep `def name(` on the same line as the function name. Keep `) -> ReturnType:` on the same line when it fits.
- Avoid trailing commas in function signatures and function calls unless multiline formatting is intentional.
- Multiline formatting is acceptable when arguments are being actively toggled or edited, when structured literals such as dictionaries read better expanded, or when expanded formatting materially improves maintainability.
