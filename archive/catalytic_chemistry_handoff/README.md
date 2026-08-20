# Catalytic-chemistry handoff

This folder contains the first fold-independent, substrate-conditioned
reaction-state prototype. It transferred experimental hydrolyzed beta-lactam
states into predicted metal/donor frames and evaluated donor fit, clashes, and
pocket contacts.

The experiment was rejected for this repository because related
metallohydrolases frequently supported similar first-shell geometry, producing
insufficient specificity. It should not be revived by merely adding a larger
classifier or more templates.

A new catalytic-chemistry project should start from the failure described in
`reports/catalytic_feasibility_no_go.md` and explicitly model information that
was absent here: catalytic water, metal electronic state, substrate carbonyl
orientation, second-shell electrostatics, proton transfer, loop dynamics, and
product release. It must have its own repository, hypotheses, datasets, and
validation gates.

The shared `ReactionTemplate` and rigid-alignment implementation remains in
`scripts/catalytic_feasibility.py` in the parent repository because the final
B1 pharmacophore imports those low-level geometry primitives.
