<a id="readme-top"></a>

<p align="center">
  <img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Swarm_2.png" alt="Swarm" width="62%" />
</p>

<h1 align="center">Swarm Champions</h1>

<p align="center">
  <b>Every model that has ever taken a crown on Swarm, published in full.</b><br/>
  <i>The source, the weights, and the score it was crowned with. Fork one and beat it.</i>
</p>

<p align="center">
  <a href="https://discord.gg/8dPqPDw7GC"><img alt="Discord" src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white" /></a>
  <a href="https://x.com/SwarmSubnet"><img alt="X" src="https://img.shields.io/badge/X-Follow-111111?style=flat-square&logo=x&logoColor=white" /></a>
  <a href="https://swarm124.com"><img alt="Website" src="https://img.shields.io/badge/swarm124.com-visit-F5D400?style=flat-square&labelColor=111111" /></a>
</p>

<p align="center">
  <a href="https://swarm124.com/benchmark"><img alt="Leaderboard" src="https://img.shields.io/badge/Leaderboard-Live-111111?style=for-the-badge" /></a>
  &nbsp;
  <a href="https://github.com/swarm-subnet/swarm/blob/main/miner/docs/miner.md"><img alt="Start Training" src="https://img.shields.io/badge/Start%20Training-Miner%20Guide-F5D400?style=for-the-badge" /></a>
</p>

---

<!-- ABOUT -->
## What Is This Repository

[Swarm](https://swarm124.com) is Bittensor subnet 124: an open arena where anyone can train a drone
pilot and prove it against the world. Each mission has one reigning champion, and the champion is
paid for as long as it holds the crown.

This repository is where the champions live. The Swarm backend writes here the moment a model is
promoted, and nothing is ever removed, so it is the complete lineage of every crown the subnet has
had. Fork any of them. Improving a published champion is how most crowns are won.

https://github.com/user-attachments/assets/ee579a55-5eb2-4f6c-83db-4b1a223b9bb2

<p align="center">
  <sub><b>Search and Rescue.</b> The mission Swarm is built around: teaching a drone to find people.</sub>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- MISSIONS -->
## The Missions

Swarm is five separate competitions, each with its own champion and its own share of the rewards.
Every one of them has a folder here.

| Mission | What the drone has to do | Reward share |
|---------|--------------------------|:------------:|
| **[Office Interceptor](https://github.com/swarm-subnet/swarm/blob/main/docs/families/office_interceptor.md)** | Hunt a target drone inside a fixed office and catch it before time runs out | <img src="https://img.shields.io/badge/30%25-F5D400?style=flat-square" alt="30%" /> |
| **[Swarm Autopilot](https://github.com/swarm-subnet/swarm/blob/main/docs/families/swarm_autopilot.md)** | Land a whole team of drones, fast and without collisions | <img src="https://img.shields.io/badge/20%25-F5D400?style=flat-square" alt="20%" /> |
| **[Swarm Search and Rescue](https://github.com/swarm-subnet/swarm/blob/main/docs/families/swarm_sar.md)** | Send a team of drones to sweep the area and find the victim together | <img src="https://img.shields.io/badge/20%25-F5D400?style=flat-square" alt="20%" /> |
| **[Autopilot](https://github.com/swarm-subnet/swarm/blob/main/docs/families/autopilot.md)** | Cross the world, find the landing pad, and touch down clean | <img src="https://img.shields.io/badge/15%25-F5D400?style=flat-square" alt="15%" /> |
| **[Search and Rescue](https://github.com/swarm-subnet/swarm/blob/main/docs/families/search_and_rescue.md)** | Find a lost person and hold a steady hover above them | <img src="https://img.shields.io/badge/15%25-F5D400?style=flat-square" alt="15%" /> |

<p align="center"><sub><b>Solved:</b> <a href="https://github.com/swarm-subnet/swarm/blob/main/docs/families/interceptor.md">Interceptor</a> — open-terrain pursuit, cleared by its champion and closed. The crown is final and the winning model is preserved here.</sub></p>

https://github.com/user-attachments/assets/a16e9453-663c-4483-a3b8-160c412fd3e7

<p align="center">
  <sub><b>Air-to-air pursuit,</b> from a real benchmark run: close the gap and catch a fleeing drone before the clock runs out.</sub>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- LAYOUT -->
## Layout

[`CHAMPIONS.md`](CHAMPIONS.md) lists the reigning champion of every mission. One folder per
mission, with its own README listing every crown, and one folder per crown inside it:

```
CHAMPIONS.md                       the reigning champion of every mission
cf_autopilot/
  README.md                        the mission and its full lineage, newest first
  uid18/
    CHAMPION.md
    model/
      drone_agent.py
      ... model weights ...
      swarm_policy_contract.json
cf_search_and_rescue/
cf_swarm_autopilot/
cf_swarm_sar/
cf_interceptor/                    archived: the crown is final
cf_interceptor_office/
```

The crown folder is named after the miner's UID; a UID that takes a second crown in the same mission
gets `uid18-2`, so no crown ever replaces another. The crowning time is in the lineage table and in
the commit. `model/` is the contents of the submitted archive and nothing else. `CHAMPION.md`
sits beside it and records the mission, the miner's hotkey and UID, the score and the per-metric
breakdown behind it, the crowning time, and the full digest of the archive. All of these pages are
written by the backend on each crown.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- RELEASES -->
## Releases

Each crown also has a GitHub Release, tagged `<family>-uid<uid>-<hash8>`, with the exact
`submission.zip` the validators evaluated attached. Its SHA-256 matches the digest in `CHAMPION.md`
and the digest the miner committed on-chain, so anyone can check that what is published is what was
scored.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- HOW -->
## How a Model Gets Here

Miners submit privately: the model goes to the Swarm backend, where it stays unpublished while it is
evaluated. Only a model that beats the reigning champion's score by the required margin is crowned,
and only a crowned model is published. Everything else stays private, forever.

The rules of the competition, the scoring, and how emissions are split among a mission's recent
kings are documented in the [swarm](https://github.com/swarm-subnet/swarm) repository, in particular
[`docs/king_of_the_hill.md`](https://github.com/swarm-subnet/swarm/blob/main/docs/king_of_the_hill.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- LICENSE -->
## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <img src="https://raw.githubusercontent.com/swarm-subnet/swarm/main/swarm/assets/Swarm.png" alt="Swarm" width="60">
</p>

<p align="center">
  <b><a href="https://swarm124.com">Swarm</a>: where AI learns to fly.</b><br/>
  <sub>Subnet 124 on <a href="https://bittensor.com">Bittensor</a></sub>
</p>
