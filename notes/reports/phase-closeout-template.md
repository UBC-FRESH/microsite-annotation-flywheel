# Phase close-out report: <phase ID and name>

Phase: <P<n> name>
Parent issue: #<number>
Pull request: #<number>, `<branch>` into `main`
Evidence head: <commit>, the head that the gates and the CI runs below cover
Report commit: the commit that adds this report. Its CI run is posted on the pull request.
Date: <YYYY-MM-DD>
Status: queued for approval. Not merged.

## Recommendation

Approve these actions. The agent carries them out in this order, and reads back each result.

1. Squash-merge pull request #<number>, and delete the branch `<branch>`.
2. Close parent issue #<number> with a closeout comment that cites the merge commit.
3. Launch <next phase>: create its branch, set the phase statuses in `ROADMAP.md`, create its child issues as sub-issues, and start its first task.

Settings changes, each with its exact command. Each one runs only after its own approval.

- <none, or the command>

## 1. Objectives and acceptance

| Criterion | Evidence | Status |
| --- | --- | --- |
| <criterion from the roadmap> | <commit, run ID, or command output> | met, met with a stated limit, or not met |

## 2. Workflow compliance

- Parent issue before the branch: <timestamps>
- Branch from `main`: <merge-base commit>
- Commit messages that start with task IDs: <commits that do, out of the total>
- Gates for each commit: section 3
- Roadmap and changelog in step: <the commits that changed them, and any that did not>
- Child issues closed with closeout evidence: <list>
- Approval gates: <each action taken, and the approval that covered it>

## 3. Gate evidence

<For each commit: the gates that it could run, and what they reported.>

## 4. CI evidence

| Run ID | Head commit | Conclusion | Notes |
| --- | --- | --- | --- |
| <run ID> | <commit> | success or failure | <cause and fix, for a failed run> |

## 5. Deviations

| Deviation | Cause | Remedy |
| --- | --- | --- |
| <what happened> | <why it happened> | <what was done, with its commit> |

## 6. Next phase

<The goal, the tasks, the acceptance criteria, the child issues, and a recommended default for each open choice, with its reason.>

## 7. Decision

- [ ] Approve as written.
- [ ] Approve with the changes stated here: <changes>.
