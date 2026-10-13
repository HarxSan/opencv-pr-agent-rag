# OpenCV PR-Agent RAG

A bot that reviews OpenCV pull requests when someone comments a command such as `/review`. It runs [Qodo PR-Agent](https://github.com/The-PR-Agent/pr-agent), adds code from the OpenCV source tree that relates to the pull request, and sends both to a fine-tuned Qwen 2.5 Coder 7B model. The related code comes from a [Qdrant](https://qdrant.tech) index of OpenCV that this repository also builds.

This repository is the commenting side: the webhook server, the indexer and the Docker setup. The code that collected the training data and ran the fine-tune is not published here.

Review quality has not been measured. There is no test set, no comparison with other models and no scoring of the reviews in this repository. Treat the output as suggestions that a person still has to read.

## How a review runs

1. Someone comments a command at the start of a comment on a pull request, for example `/review`.
2. GitHub sends an `issue_comment` event to `POST /webhook`. The server checks the `X-Hub-Signature-256` header against `GITHUB_WEBHOOK_SECRET` and answers 401 on a mismatch. If the secret is not set, it answers 503 and does nothing else.
3. The server answers 202 and does the rest in a background thread. It only acts on newly created comments on pull requests that start with a known command.
4. It puts an "eyes" reaction on the comment, then fetches the PR title, description, changed file names and diff from the GitHub API. The file list stops at `MAX_FILES_PER_PR` and the diff is cut at `MAX_DIFF_SIZE`.
5. For `/review`, `/improve` and `/ask`, it searches the Qdrant index. It builds up to 8 short queries from the title, the OpenCV terms in the title and description, patterns found in the diff (such as `cv::Mat` names and `CV_` macros) and the changed file names. Each query is embedded with `BAAI/bge-base-en-v1.5`. Chunks scoring below `RAG_MIN_SCORE` are dropped, the rest are reranked, and the top `RAG_TOP_K` are formatted as context. If the index is empty or Qdrant is down, the command still runs without context.
6. It writes temporary PR-Agent settings that contain OpenCV-specific review instructions and the retrieved code, then runs `python -m pr_agent.cli --pr_url=<url> <command>`.
7. PR-Agent calls the model through the OpenAI-style endpoint in `LIGHTNING_AI_ENDPOINT` and posts the result on the pull request itself, using the same GitHub token.
8. The server changes the reaction to "rocket" on success. On failure it adds a "confused" reaction and posts the error text as a comment.

The code that gets searched is whatever OpenCV checkout you indexed. It is not the branch the pull request comes from.

## The model and the training data

- **Model:** [nareshmlx/code-reviewer-opencv-harxsan-v2](https://huggingface.co/nareshmlx/code-reviewer-opencv-harxsan-v2) on Hugging Face. Its card says it was fine-tuned from `unsloth/qwen2.5-coder-7b-instruct-bnb-4bit` (a 4-bit copy of [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)), lists the tags `sft`, `trl` and `unsloth`, and names the Apache-2.0 license. It was uploaded on 2025-11-24 to the account `nareshmlx`.
- **Who did what:** Harii Sankar collected and prepared the data and ran the fine-tune with [Unsloth](https://github.com/unslothai/unsloth) (LoRA / QLoRA). A colleague was also involved.
- **Data:** The raw collection was more than 100,000 review comments from OpenCV's pull request history. They were filtered and processed down to 14,203 samples, published as [harxsan/opencv-pr-data](https://huggingface.co/datasets/harxsan/opencv-pr-data). Each row pairs a pull request's code context with the original maintainer comment: `instruction` (a fixed review prompt), `input` (the PR title and description, file path, language, diff, the target hunk and line, and an extracted function context called `enhanced_context`) and `output` (the maintainer's comment). A separate Gemini step turned comments into structured JSON reviews; those are not published.
- **Not published:** the collection scripts, the Gemini step, the training script and any evaluation results.

The repository does not include any code for serving the model. It expects an endpoint that answers OpenAI-style `/v1/chat/completions` requests.

## What is in this repository

```
docker-compose.yml        Qdrant, the indexer (only runs on request) and the webhook server
.env.example              Template for .env (copy it, then fill it in)
setup.sh                  Creates the data/ and secrets/ folders and fills in two blank .env lines
test.sh                   Checks that Qdrant, the server, RAG and the model endpoint answer
indexer/
  Dockerfile
  indexer.py              Reads the OpenCV checkout, chunks it, embeds it, writes it to Qdrant
  requirements.txt
webhook-server/
  Dockerfile              Installs PR-Agent from its main branch
  server.py               Flask app: /health, /webhook, /context, /test
  pr_agent_runner.py      Parses the command, builds the PR-Agent settings, runs PR-Agent
  rag_retriever.py        Query building, Qdrant search, reranking, formatting
  config.py               Reads every setting from environment variables
  requirements.txt
docs/
  GITHUB_APP_SETUP.md     GitHub App walk-through (see Known gaps)
  TROUBLESHOOTING.md
```

**Indexer.** It reads `.cpp`, `.hpp`, `.h`, `.c`, `.cc`, `.cxx`, `.py`, `.cu` and `.cuh` files and skips folders named `build`, `3rdparty`, `cmake`, `doc`, `data`, `samples`, `test`, `tests`, `perf`, `android`, `ios`, `platforms`, `apps`, `java`, `js` and `objc`, plus folders that start with a dot. C++ and Python files are split by function and class with tree-sitter. If that fails, it falls back to fixed pieces of 80 lines with 15 lines of overlap. Embedding runs on the CPU. Each run deletes the existing collection and builds it again.

## Setup

You need Docker with Compose, `git`, `curl`, [ngrok](https://ngrok.com) (or any other public HTTPS address that forwards to port 5000), a GitHub token, and a model endpoint. The docs below use `docker-compose`. If you have Compose v2, `docker compose` works the same way.

1. **Get the code.**
   ```bash
   git clone https://github.com/HarxSan/opencv-pr-agent-rag.git
   git clone https://github.com/opencv/opencv.git
   cd opencv-pr-agent-rag
   mkdir -p data/qdrant data/logs data/cache secrets
   ```

2. **Create `.env`.** Copy the template and fill it in. `./setup.sh` does the copy for you and also fills in `GITHUB_WEBHOOK_SECRET` and `QDRANT_API_KEY` when they are empty (it needs `docker` and `docker-compose` on the path). Set `OPENCV_REPO_PATH` before running any `docker-compose` command, because `docker-compose.yml` marks it as required.
   ```bash
   cp .env.example .env
   ```
   These are the values to set:
   ```bash
   GITHUB_USER_TOKEN=github_pat_xxxx
   GITHUB_WEBHOOK_SECRET=<output of: openssl rand -hex 32>
   ADMIN_TOKEN=<output of: openssl rand -hex 32>
   OPENCV_REPO_PATH=/absolute/path/to/opencv
   LIGHTNING_AI_ENDPOINT=https://your-model-host.example
   LIGHTNING_AI_MODEL_NAME=nareshmlx/code-reviewer-opencv-harxsan-v2
   LIGHTNING_AI_API_KEY=not-used
   ```
   `GITHUB_WEBHOOK_SECRET` and `LIGHTNING_AI_ENDPOINT` are required: with no secret, `/webhook` answers 503, and with no endpoint, `/health` reports a problem. `ADMIN_TOKEN` is needed for `/test` and `/context`; if it is not set, those two routes answer 404. The token needs Contents (read), Issues (read and write) and Pull requests (read and write) on the repository you want reviewed. Use a token. The server's own calls to GitHub do not work without one (see Known gaps). If `setup.sh` set `QDRANT_API_KEY`, the Qdrant `curl` calls below need `-H "api-key: $QDRANT_API_KEY"` (see Settings). The other settings are listed further down.

3. **Start Qdrant.**
   ```bash
   docker-compose up -d qdrant
   curl http://localhost:6333/readyz
   ```

4. **Index OpenCV.** This reads the checkout from `OPENCV_REPO_PATH` and can take a while on a CPU. No timing has been recorded.
   ```bash
   docker-compose --profile indexer run indexer
   curl -s http://localhost:6333/collections/opencv_codebase | jq .result.points_count
   ```
   The second command should print a number above zero.

5. **Start the webhook server.** The first start downloads the embedding model, so give it a few minutes.
   ```bash
   docker-compose up -d pr-agent-rag
   curl http://localhost:5000/health
   ```
   `/health` answers 200 with `"status":"healthy"` when Qdrant has points and a model endpoint is set. Otherwise it answers 503 with `"status":"degraded"`. If `LIGHTNING_AI_ENDPOINT` is empty, the `model` part shows `"ok":false` and `"endpoint":"not configured"`. `/health` needs no token.

6. **Check everything.** `./test.sh` reads `.env` (so it picks up `ADMIN_TOKEN` for its `/context` check), runs eight checks and prints how many passed.
   ```bash
   ./test.sh
   ```

7. **Expose port 5000 and add the webhook.**
   ```bash
   ngrok http 5000
   ```
   In the GitHub repository, go to Settings, Webhooks, Add webhook. Payload URL `https://<your-ngrok-address>/webhook`, content type `application/json`, secret the same as `GITHUB_WEBHOOK_SECRET`, and the event "Issue comments". The server also accepts "Pull requests" events but only logs them.

8. **Try it.** Comment `/review` on a pull request. Other commands are listed below.

Day to day:

```bash
docker-compose logs -f pr-agent-rag     # watch the server
docker-compose down                     # stop everything
docker-compose build --no-cache pr-agent-rag   # rebuild after code changes
docker-compose --profile indexer run indexer   # re-index (deletes the old collection first)
```

You can also try a review without the webhook. This needs `ADMIN_TOKEN` (here read from your shell, for example after `source .env`):

```bash
curl -X POST http://localhost:5000/test \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: $ADMIN_TOKEN" \
  -d '{"pr_url": "https://github.com/OWNER/REPO/pull/1", "command": "review"}'
```

## Settings

All settings are environment variables. Docker Compose passes them from `.env`. Where `docker-compose.yml` and `config.py` disagree, the value Docker uses is the compose one.

| Variable | Default | What it does |
|---|---|---|
| `GITHUB_USER_TOKEN` | none | GitHub token used for all GitHub calls |
| `GITHUB_WEBHOOK_SECRET` | none, required | Webhook signature secret. If empty, `/webhook` answers 503. |
| `ADMIN_TOKEN` | none | Token for `POST /test` and `POST /context`, sent as the `X-Admin-Token` header. If empty, those routes answer 404. |
| `OPENCV_REPO_PATH` | none, required by Compose | Path to the OpenCV checkout the indexer reads |
| `LIGHTNING_AI_ENDPOINT` | none, required | Base address of the model server. If empty, `/health` reports it as not configured. |
| `LIGHTNING_AI_MODEL_NAME` | `nareshmlx/code-reviewer-opencv-harxsan-v2` | Model name sent in each request |
| `LIGHTNING_AI_API_KEY` | `not-used` | API key sent to the model server |
| `MODEL_MAX_TOKENS` | 32000 | Token limit given to PR-Agent |
| `MODEL_TEMPERATURE` | 0.1 in Compose, 0.3 in `config.py` | Sampling temperature |
| `QDRANT_COLLECTION_NAME` | `opencv_codebase` | Collection the indexer writes and the server reads |
| `QDRANT_API_KEY` | none | Optional Qdrant key. If you set it, the `curl` calls to `/collections/...` in this README need `-H "api-key: $QDRANT_API_KEY"`. |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Must be the same for the indexer and the server |
| `RAG_TOP_K` | 15 | Number of code chunks kept |
| `RAG_MIN_SCORE` | 0.4 | Lowest search score a chunk may have |
| `RAG_MAX_CONTEXT_TOKENS` | 16000 in Compose, 12000 in `config.py` | Size cap on the context (counted as 4 characters per token) |
| `RAG_ENABLED_COMMANDS` | `review,improve,ask` | Commands that get code context |
| `MAX_DIFF_SIZE` | 10485760 (10 MB) | The diff the server fetches for searching is cut at this size |
| `MAX_FILES_PER_PR` | 500 | The server reads at most this many changed files |
| `REQUEST_TIMEOUT` | 600 | Seconds before a PR-Agent run is stopped |
| `EMBEDDING_BATCH_SIZE`, `UPLOAD_BATCH_SIZE`, `PARALLEL_WORKERS`, `INDEXING_THRESHOLD`, `MAX_INDEXING_THREADS` | 128, 500, 12, 10000, 4 | Indexer tuning |

`GITHUB_REPO`, `GITHUB_APP_ID` and `GITHUB_APP_PRIVATE_KEY_PATH` also exist. `GITHUB_REPO` has no default and is read but nothing uses it. The two App settings are covered under Known gaps. `LOG_LEVEL` is passed in but the server sets its log level to INFO when it starts.

## Commands

A comment has to start with the command. Anything after `/ask` is passed on as the question.

| Command | What it does | Gets code context |
|---|---|---|
| `/review` | Reviews the pull request | yes |
| `/improve` | Suggests code improvements | yes |
| `/ask <question>` | Answers a question about the pull request | yes |
| `/describe` | Writes a description | no |
| `/update_changelog` | Updates the changelog | no |
| `/add_docs` | Adds documentation | no |
| `/test` | Generates tests | no |
| `/help` | Shows PR-Agent help | no |
| `/config` | Shows PR-Agent settings | no |

## HTTP endpoints

- `GET /health`: status of Qdrant, the model settings and GitHub login
- `POST /webhook`: receives GitHub events
- `POST /context`: runs only the search. Send `title`, `description`, `changed_files` and `diff`. Needs the `X-Admin-Token` header.
- `POST /test`: runs a command on a pull request. Send `pr_url` and `command`. Needs the `X-Admin-Token` header.

## Troubleshooting

**`/health` says degraded, or the container shows unhealthy.** The RAG part reports "Collection not available or empty". Run the indexer (setup step 4) and check the point count.

**Qdrant checks.**
```bash
docker-compose ps qdrant
docker-compose logs qdrant
curl http://localhost:6333/readyz
curl -s http://localhost:6333/collections/opencv_codebase | jq .result.points_count
```
`/readyz` works without a key. If `QDRANT_API_KEY` is set, add `-H "api-key: $QDRANT_API_KEY"` to the last command. `./test.sh` does not send the key, so its collection check fails when a key is set.

**401 "Invalid signature".** The secret in the GitHub webhook and `GITHUB_WEBHOOK_SECRET` differ, or one has a trailing space. A 503 on `/webhook` means `GITHUB_WEBHOOK_SECRET` is empty in the container.

**404 or 403 on `/test` or `/context`.** 404 means `ADMIN_TOKEN` is not set in the container. 403 means the `X-Admin-Token` header does not match it.

**Nothing happens when I comment.** Open the webhook's "Recent Deliveries" page on GitHub and look at the response. Then check `docker-compose logs -f pr-agent-rag`. The comment has to be new, on a pull request, and begin with the command.

**"Failed to fetch PR information".** The token is missing, expired or lacks access. Test it directly:
```bash
curl -H "Authorization: token $GITHUB_USER_TOKEN" https://api.github.com/repos/OWNER/REPO/pulls/1
```

**Model errors.**
```bash
curl -X POST "$LIGHTNING_AI_ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$LIGHTNING_AI_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":10}"
```
If the endpoint wants a key, add `-H "Authorization: Bearer $LIGHTNING_AI_API_KEY"`.

**The review shows no code from OpenCV.** Check that the collection has points, try a lower `RAG_MIN_SCORE`, and make sure the indexer and the server use the same `EMBEDDING_MODEL`. You can test the search alone:
```bash
curl -X POST http://localhost:5000/context \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: $ADMIN_TOKEN" \
  -d '{"title":"Fix Mat issue","changed_files":["modules/core/src/matrix.cpp"],"diff":""}'
```

**Timeouts on large pull requests.** Raise `REQUEST_TIMEOUT`. PR-Agent fetches and clips the diff itself (`large_patch_policy = "clip"` in `pr_agent_runner.py`), so `MAX_DIFF_SIZE` only limits the copy the server uses for searching.

**The indexer uses too much of the machine.** `docker-compose.yml` lets it use up to 12 CPUs and 8 GB. Lower the limits in its `deploy` block.

## Known gaps

- **No measurement of review quality.** There are no tests, no benchmark and no results file.
- **PR-Agent is not pinned.** `webhook-server/Dockerfile` installs it from the `main` branch, so a rebuild can bring in different code. `torch` and `transformers` are also only given a minimum version in `webhook-server/requirements.txt`.
- **GitHub App login is only partly wired.** The server's own GitHub calls (fetching the PR, reactions, error comments) send only `GITHUB_USER_TOKEN`. App settings are passed to PR-Agent and nowhere else, and the code that would turn an App key into an installation token is not in this repository. `docs/GITHUB_APP_SETUP.md` describes the App route in more detail than the code supports. Use a token.
- **No sender check.** The server does not look at who wrote a comment or which repository sent it. Any signed event from a repository that has the webhook is handled.
- **Two commands at the same moment can interfere.** `PRAgentRunner` keeps one temporary settings folder per runner object and deletes the previous one when a new command starts. By reading the code, two commands arriving together in the same worker can remove each other's folder. This has not been tested.
- **Defaults disagree.** Temperature is 0.1 in Compose and 0.3 in `config.py`. Context tokens are 16000 in Compose and 12000 in `config.py`.
- **`test.sh` and a Qdrant key.** The collection check does not send the key.
- **Indexing runs on the CPU** and rebuilds the whole collection every time.
- **No license file.**
- **Public address needed.** GitHub has to reach port 5000. The docs use ngrok, whose address changes when the tunnel restarts. For anything longer-lived, use a fixed HTTPS address and a Qdrant volume you back up.
- **The model lives on another Hugging Face account** (`nareshmlx`). If that page changes, the default model name here stops working.
