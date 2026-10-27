# How we work in this repo

The aim: three people working in parallel without overwriting each other, and `main` always works.

## The rules

1. **Never commit straight to `main`.** Work on a branch and merge through a pull request (PR).
2. **Stay in your own folders** (see the ownership table in the README). To change someone else's folder, open a PR and tag them.
3. **Shared files need two approvals:** `reasoning/data.py`, `prompts.py`, `answer.py`, `stats.py`, `data/splits/` and `tests/`. A change there changes everyone's results.
4. **Never commit large outputs:** `runs/`, `ckpt/`, SLURM logs (`slurm-*.out`), model weights, raw teacher output. They are gitignored. Keep them on the cluster.
5. **Tests must pass** before you merge. GitHub runs them on every push, and you can run them yourself with `python -m pytest tests`.

## Daily workflow

```bash
# 1. Start from the latest main
git checkout main
git pull

# 2. Make a branch for one task, prefixed with your letter
git checkout -b a/teacher-generation      # Person A
git checkout -b b/sft-script              # Person B
git checkout -b c/cluster-baselines       # Person C

# 3. Work, then commit small steps with clear messages
git add reasoning/teacher/generate.py
git commit -m "Teacher generation with vLLM for standard and long styles"

# 4. Push the branch and open a pull request on GitHub
git push -u origin a/teacher-generation
```

On GitHub, click **Compare & pull request**. Write what changed and how you checked it, then request a review from one teammate (or both, for shared files). The reviewer reads the change, asks questions, and approves. Then click **Squash and merge** and delete the branch.

## Keeping your branch up to date

If `main` moved while you were working:

```bash
git checkout main && git pull
git checkout a/teacher-generation
git merge main            # fix any conflicts, then commit
```

## Commit messages

Say what changed and why, in one line: "Add rejection-sampling filter using answer.is_correct". Avoid "update" or "fix stuff".

## Results that go into the report

Copy the small summary files into `results/tables/` and the final plots into `results/figures/`, and commit them with the command that produced them in the commit message. Anyone should be able to trace every number in the report back to a committed file.

## AI use

We are allowed to use AI tools, but the report needs an AI Use Statement. When AI helps with a piece of code, say so in the PR description (for example "drafted with Claude, reviewed and tested by me"). That makes the statement easy to write at the end.
