# jav67

Ranks a set of labelled options against a phrase by cosine similarity of embeddings from a locally hosted Ollama model, through the agno framework. Default use: classify the emotion of a sentence as Anger, Fear or Joy.

## Run

Command line:

```bash
uv run main.py "I've had enough!"
```

Web UI:

```bash
uv run streamlit run app.py
```

The UI opens at http://localhost:8501. Edit the phrase, the instruction and the options table, press Run, then move the sharpness slider to see confidence change without re-embedding.

## Requirements

| Need | Value |
|---|---|
| Python | 3.14 or newer, managed by [uv](https://docs.astral.sh/uv/) |
| Ollama | running at http://localhost:11434 |
| Model | `qwen3-embedding:0.6b`, 1024 dimensions |

Install the model once:

```bash
ollama pull qwen3-embedding:0.6b
```

`uv run` installs the Python dependencies (agno, ollama, pydantic, streamlit) on first use.

## Keep Ollama warm

Ollama unloads a model after 5 minutes without requests. The next request then pays 1.5 to 2.5 seconds to reload it. A warm embedding call takes 12 to 20 ms.

Pin the model for a fixed time with one request. Later requests do not shorten the pin.

```bash
curl -s http://localhost:11434/api/embed \
  -d '{"model":"qwen3-embedding:0.6b","input":"warmup","keep_alive":"1h"}' > /dev/null
```

`keep_alive` accepts durations such as `30m`, `1h`, `2h30m`, or `-1` to keep the model loaded until Ollama restarts. Check the expiry in the `UNTIL` column of `ollama ps`.

To make every model stay loaded by default on a systemd install, run `sudo systemctl edit ollama`, add the lines below, then `sudo systemctl restart ollama`:

```ini
[Service]
Environment="OLLAMA_KEEP_ALIVE=-1"
```

## Define phrases and options

The phrase is the positional argument. Options are given as `Label` or `Label: description`. A description sharpens the match, so prefer it.

```bash
uv run main.py "Something is moving in the dark outside"
uv run main.py "What a day" --options "Sadness: feeling down or hopeless" "Relief: tension has passed" Pride
uv run main.py "I've had enough!" --instruction "Classify the mood of this customer message"
uv run main.py "I've had enough!" --no-instruction
uv run main.py "I've had enough!" --sharpness 20 --json
```

At least two options are required. To change the defaults for both the CLI and the UI, edit `DEFAULT_OPTIONS` and `DEFAULT_INSTRUCTION` in `main.py`.

In the UI, the options table accepts new rows. Leave the description empty to embed the bare label.

### Instruction

Qwen3 embedding models expect the query wrapped as `Instruct: <task>\nQuery: <text>`. Options are embedded bare. On the three test phrases in this repo the instruction raised the gap between the top two labels from about 0.02 to 0.10 or more. Other models ignore or mishandle this prefix, so pass `--no-instruction` when switching models.

## Output

| Column | Meaning |
|---|---|
| similarity | cosine similarity between phrase and option, -1 to 1 |
| distance | 1 minus similarity |
| scaled | similarity min-max scaled across options: best 1.0, worst 0.0 |
| confidence | softmax of sharpness times similarity, sums to 1 across options |
| embed ms | time to embed that option |

Confidence uses raw similarities, not scaled ones. A near tie between the top two stays near even at any sharpness. Sharpness 0 gives an even split. On a 0.13 similarity gap, sharpness 10 gives about 72% and sharpness 20 about 92% for the best option.

Timings are split into phrase embedding, options embedding, scoring and total. Options embedding is the largest part and is constant for a fixed option set.

## Failures

| Symptom | Cause |
|---|---|
| `No embedding returned for ...` | Model not pulled, or `--dimensions` does not match the model. agno returns an empty vector on a size mismatch instead of raising. |
| Connection refused on port 11434 | Ollama is not running. Start it with `ollama serve` or `systemctl start ollama`. |
| `List should have at least 2 items` | Fewer than two options given. |
| First run takes over a second | Model was unloaded. See Keep Ollama warm. |

Other pulled models and their sizes: `nomic-embed-text` 768, `mxbai-embed-large` 1024, `bge-m3` 1024. Pass `--model` and `--dimensions` together.

## Files

- `main.py`: pydantic models, embedding, scoring and the CLI.
- `app.py`: Streamlit UI. Imports everything from `main.py`.
