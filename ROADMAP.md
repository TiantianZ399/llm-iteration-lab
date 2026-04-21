# Roadmap

This roadmap reflects the repository's current stage: a public research-engineering scaffold with the first implementation direction still being stabilized.

## Phase 1: Public repository setup

- establish a clear README, license, contribution guide, and roadmap
- define the repository's umbrella direction without overclaiming scope or maturity
- document the first implementation direction at a level that is useful for public development

## Phase 2: Problem definition and architecture sketch

- clarify the boundaries of the first vertical
- define what counts as evaluation, diagnosis, refinement, and iteration inside the project
- sketch an initial modular structure for future implementation

## Phase 3: First vertical prototype

- build a minimal prototype for the initial evaluation-diagnosis-refinement loop
- keep the prototype narrow, inspectable, and honest about limitations
- start identifying the right artifact formats and interfaces for repeated iteration

## Phase 4: Evaluation and iteration tooling

- add tooling to compare successive runs and revisions
- improve failure recording, diagnosis traceability, and iteration history
- make it easier to rerun bounded workflows under the same assumptions

## Phase 5: Examples and reproducibility

- add small public examples once the first vertical is stable enough to show
- document expected inputs, outputs, and workflow assumptions
- tighten reproducibility around the initial workflow

## Phase 6: Broader modularization

- expand beyond the first vertical when the core loop is clear enough
- separate reusable components from task-specific logic
- prepare the repository for broader open-source collaboration
