# UID154 V5.1.5.6 dynamic-office candidate

This is a conservative adaptation of the UID154 Phase-4 candidate for the
V5.1.5.5 office distribution (included in Swarm V5.1.5.6).

The action-path changes are deliberately small and causal:

* take off vertically with horizontal and yaw sticks held at zero until the
  drone has passed both the 2.0 s startup window and 1.25 m altitude;
* disable the pre-local launch planner and TTC shield because both query
  `office_grid.npz`, whose 41 furniture placements no longer match an episode;
* use the calibrated 0.035 m camera-forward extrinsic in the primary target
  tracker.

Passive actor features remain enabled. A paired ablation that zeroed the old
map feature block created a new obstacle collision, so removing it would not be
a safe compatibility change for the frozen actor. The old close-range
consensus module likewise retains its jointly calibrated 0.13 m internal
coordinate convention; changing that constant alone recreated a known tilt.
These two components must be replaced atomically with a retrained actor or
estimator rather than partially patched.

The previous temporal RGB risk model remains unchanged. It is bounded to the
original short launch window and is not treated as a general solution for
random furniture.

Environment variables retain the old ablation controls. In particular,
`OP_PRE_ON=1`, `OP_SH_ON=1`, or the four `OP_TO_*` variables restore individual
legacy paths for local diagnosis.
