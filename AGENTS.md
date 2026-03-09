## Python Formatting

- Prefer single-line function signatures and function calls when they fit within 99 columns.
- Keep compact signatures and calls compact. Do not split arguments one-per-line unless
  line length or readability requires it.
- If multiline formatting is needed, group multiple arguments per line when practical
  instead of using one argument per line.
- Keep `def name(` or `callee(` on the first line and pack that line as far as the
  99-column limit allows before wrapping.
- When wrapping signatures or calls, prefer aligned continuation lines over hanging
  closing-paren layouts. Keep the closing `)` and `-> ReturnType` on the last
  argument line when they fit.
- Avoid adding trailing commas solely to force a vertically expanded layout.
- Keep string literals on one physical line when they fit within 99 columns. Avoid
  splitting one message into adjacent string literals unless needed for line length
  or readability.
- Multiline formatting is acceptable when arguments are being actively toggled or
  edited, when structured literals such as dictionaries read better expanded, or
  when expanded formatting materially improves maintainability.
