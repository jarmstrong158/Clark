"""Clark operations dashboard.

A web UI for running plans / what-ifs / comparisons / sweeps / calendar
checks against `clark serve` via forms. Distinct from `clark dashboard`
(the training-metrics observability dashboard). This is for the
warehouse operator: the audience that doesn't want to type into a
chat box and have a language model maybe-correctly translate them.

Launches in a separate HTTP server (stdlib) so it can be opened, used,
and closed without affecting the chat, the wizard, or the inference
server.

Run:  clark ops              # port 8092
      clark ops --port 9000
"""
