# RUNBOOK: operating the annotation flywheel

This runbook is for the person who runs the loop. You do not need to read or write any code.
You run four commands and do one thing in Roboflow between them.

## What the loop does

Each round, the flywheel:

1. downloads the annotations you have already made (**sync**),
2. trains a model to imitate them (**train**),
3. uses that model to guess the microsites on the images you have **not** yet annotated, and
   uploads those guesses as **editable pre-labels** (**prelabel**),
4. measures how much you changed the pre-labels (**compare**), and
5. shows a table of every round so far (**status**).

You correct the pre-labels in Roboflow. Correcting a guess is much faster than painting from
scratch, so each round should need less of your time.

## One-time setup

Do this once, in a terminal, in the project folder.

1. Create and fill the Python environment:

   ```bash
   python3 -m venv .venv
   . .venv/bin/activate
   pip install -r requirements-dev.txt
   ```

   On a shared server a prepared environment may already exist; ask Gregory which to use.

2. Create a file called `.env` next to `config.yaml`, containing your Roboflow key:

   ```
   ROBOFLOW_API_KEY=your_key_here
   ```

   This file is gitignored and must never be committed or pasted anywhere. The key comes from
   Roboflow under Account > Roboflow Keys.

3. Check `config.yaml`. It names the workspace, the project to train on, the project to
   pre-label, the class name, and the round number. Change the round number when you start a
   new round. Do not change anything else unless Gregory asks you to.

## Running a round

Run the commands from the project folder, with the environment active
(`. .venv/bin/activate`).

### Step 1: annotate (in Roboflow)

Finish and save the annotations you want this round to learn from, in the project named by
`train_project` in `config.yaml`.

### Step 2: sync

```bash
python scripts/loop.py sync
```

Downloads the annotated images and their masks into `_data/round-<n>/`, and writes a record of
what it saw. It prints how many images were annotated and how many are still empty. To try it on
a few images only, add `--limit 8`.

### Step 3: train

```bash
python scripts/loop.py train
```

Trains the pre-labeller on what `sync` downloaded. It prints the validation score each epoch and
stops early on its own. It writes a checkpoint and a record of the recipe and the score. On a
GPU this takes a few minutes; the server has one. To train a quick throwaway model for practice,
add `--random-init` and lower `epochs` in `config.yaml`.

### Step 4: prelabel

```bash
python scripts/loop.py prelabel
```

Runs the model over the images in the pre-label project that have **no** annotations yet, and
uploads its guesses as editable pre-labels. It prints each image it writes, and it **skips** any
image that already has an annotation. It never overwrites your work. To try it on a few images
only, add `--limit 3`.

### Step 5: correct (in Roboflow)

Open the project, review the model's pre-labels, and fix them: accept, delete, or drag the
shapes until they are right. This is the part only you can do. If nothing appears to review,
tell Gregory: the model may have found nothing to guess.

### Step 6: compare

```bash
python scripts/loop.py compare
```

Compares what was uploaded against what is now in the project, and records how much you changed.

### Step 7: status

```bash
python scripts/loop.py status
```

Prints a table of every round.

## How to tell a round is done

A round is done when you have corrected the pre-labels and `compare` runs without error. Then
bump `round` in `config.yaml` and start the next round at Step 2.

## Reading the trend table

```
ROUND  IMAGES   ADDED  DELETED  EDITED  OBJ_CHG  PIX_IOU  PIX_ADD  PIX_DEL
    1       3       0        0       0    0.000    1.000    0.000    0.000
```

- **ADDED / DELETED / EDITED**: microsites you added, deleted, or reshaped compared with the
  model's guess.
- **OBJ_CHG**: the fraction of guesses you did not leave untouched. `0.0` means you accepted
  everything.
- **PIX_IOU**: how well the guessed shape matched your correction. `1.000` is a perfect match.
- **PIX_ADD / PIX_DEL**: the fraction of microsite area you added or removed.

**When to stop:** when ADDED, DELETED, and EDITED are all small and the numbers have stopped
falling for two rounds in a row, the model is as good as this project's annotations can make it.
Send Gregory the table; he will decide with you whether to stop or start a new, harder round
(different microsite definition, or a different block).

## Safety rules (the tool follows these for you)

- Pre-labels go only to images with no annotations, with overwriting switched off. If an image
  already has an annotation, the tool skips it and says so.
- Secrets stay in `.env`; nothing secret is ever written into the repository or a report.
- Big files (images, checkpoints, predictions) stay in the data folder, never in the code
  repository.

## If something goes wrong

- **"ROBOFLOW_API_KEY is not set"**: your `.env` is missing or in the wrong folder. It must sit
  next to `config.yaml`.
- **"configuration file not found"**: you are in the wrong folder. `cd` into the project folder
  first.
- **`prelabel` uploaded nothing**: either every candidate image already has an annotation (good),
  or the model guessed nothing on the images it saw. Run `--limit 3` and watch the output.
- **A command stops with an error**: copy the last few lines and send them to Gregory. Do not
  retry a destructive command; nothing this tool does destroys annotations.
