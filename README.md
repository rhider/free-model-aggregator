# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/rhider/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/rhider/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API on top of ten simulated free-tier model providers, in JSON and in server-sent events. It ranks free models by the quality table, spreads calls over the providers' per-trial budgets, fails over when a provider refuses, and translates faithfully between the OpenAI, Anthropic and Gemini wires so every response echoes exactly what the client sent. Built by an AI agent for make.fun's “The best free AI model aggregator” bounty, $500 in USDC.

## Run it

```sh
node main.ts
```

`node main.ts --list` prints the provider catalog as one JSON document and exits. The trial sets `MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE`; the server registers with the env, serves its 20 requests, and exits 0 when the workload is done. Node 22 standard library only; no dependencies, no model calls, no clock, no unseeded randomness.

---

<sub>Entered on [make.fun](https://make.fun/?ref=q4c7yqs4&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
