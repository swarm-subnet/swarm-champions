# Swarm champions

Every model that has taken a crown on [Swarm](https://swarm124.com), Bittensor subnet 124, published
in full: the source, the weights, and the score it was crowned with. This repository is written by
the Swarm backend the moment a model is promoted, and nothing in it is ever removed, so it is the
complete lineage of every champion the subnet has had.

Fork any of them. Improving a published champion is how most crowns are won.

## Layout

One folder per challenge family, one folder per crown inside it:

```
cf_autopilot/
  2026-09-03_uid-18_a1b2c3d4/
    drone_agent.py
    ... model weights ...
    swarm_policy_contract.json
    CHAMPION.md
cf_search_and_rescue/
cf_swarm_autopilot/
cf_swarm_sar/
cf_interceptor/
cf_interceptor_office/
```

The folder name is the crowning date, the miner's UID, and the first eight characters of the
archive's SHA-256. `CHAMPION.md` records the family, the miner's hotkey and UID, the score and the
per-metric breakdown behind it, the crowning time, and the full digest of the archive.

## Releases

Each crown also has a GitHub Release, tagged `<family>-uid<uid>-<hash8>`, with the exact
`submission.zip` the validators evaluated attached. Its SHA-256 matches the digest in `CHAMPION.md`
and the digest the miner committed on-chain, so anyone can check that what is published is what was
scored.

## Who writes here

Commits are made by the subnet's publishing app, authenticated with a short-lived token minted per
run. It writes new crown folders and creates their Releases; it never modifies or deletes anything
already published. There are no manual commits.

## How a model gets here

Miners submit privately: the model goes to the Swarm backend, where it stays unpublished while it is
evaluated. Only a model that beats the reigning champion's score by the required margin is crowned,
and only a crowned model is published. Everything else stays private, forever.

The rules of the competition, the scoring, and how emissions are split among a family's recent kings
are documented in the [swarm](https://github.com/swarm-subnet/swarm) repository, in particular
[`docs/king_of_the_hill.md`](https://github.com/swarm-subnet/swarm/blob/main/docs/king_of_the_hill.md).
