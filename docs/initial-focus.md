# Initial Focus

This note captures the repository's first implementation direction.

## What the first vertical is

The initial target is a bounded workflow for iterative evaluation, diagnosis, and refinement of a concrete LLM system setting. The intent is to make improvement loops more structured and inspectable, not to imply that the project already provides a general autonomous development system.

## Intended loop

The first vertical is expected to center on a repeatable sequence such as:

1. define a task slice, objectives, and evaluation criteria
2. run the current system on that slice
3. collect failures in a structured form
4. attach diagnoses or failure hypotheses
5. propose or apply constrained refinements
6. rerun the slice and compare the next iteration against the previous one

## Why start here

This is narrow enough to be auditable and practical, while still representing the broader direction of automated development loops for LLM systems.

## Current status

This is a scope note, not a released implementation specification. Exact task definitions, interfaces, and artifact formats are still being finalized in public.
