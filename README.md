# Solari Autonomous Debugging + Agent Trajectory Evaluation

This project is basically an extension of an idea that I worked on previously around collecting high quality developer debugging traces for training coding agents which you can check out right here: https://github.com/Immrudul/agent-training-data-pipeline

The previous version looked at:

> human developer debugging → capture what they did → verify the final code → evaluate the quality of the debugging trajectory

This project flips that around a little bit:

> coding agent debugging → execute everything inside a real sandbox → capture what the agent did → independently verify the result → evaluate the trajectory

The main idea is to create an isolated environment where an AI coding agent can investigate an actual repository, reproduce failing tests, inspect code, make edits, rerun tests, and eventually produce a verified fix.

How does this connect to my previous developer trace project?

Well, that project focused on human-generated debugging data:

human developer
→ debugging actions
→ captured trajectory
→ verification
→ quality evaluation
→ training / demonstration data

But this project explores the other side of the same problem by generating agent rollouts inside controlled environments:

coding agent
→ Solari sandbox
→ debugging actions
→ tests + patches
→ deterministic verification
→ trajectory evaluation
→ rollout / preference / RL data

So instead of only thinking about how to collect examples of how good developers solve problems, this project looks at how coding agents themselves behave when given a real repository, a failing test, and tools to interact with the environment.

I think that distinction is pretty useful because it leads us to having differnt used for each pipeline. Human developer trajectories can act essentially act as a demo of desirable debugging behavior, while agent-generated trajectories can give us successes, failures, alternative approaches, and reward-labelled rollouts and also comes together as a much stronger benchmarking pipeline because of the following reason:

The snapshot experiments allow multiple agents/attempts can start from the exact same failing state:

same bug
   |
   +→ trajectory A → success → 9.8
   |
   +→ trajectory B → success → 7.4
   |
   +→ trajectory C → failure → 4.4

That starts to look a lot like the kind of data we could use for preference learning or RL for coding agents:

{
  "chosen": "trajectory_A",
  "rejected": "trajectory_B"
}

or:

{
  "trajectory": "trajectory_A",
  "tests_passed": true,
  "reward": 9.8
}

So across both projects, I’ve basically worked with both sides of the coding-agent data problem: collecting and evaluating human demonstration data, and generating and evaluating agent rollout data in reproducible environments and I think that combination is a lot more interesting than looking at either project in isolation.

Every action along the way is captured into a structured trace.

For the main E2E example, I use my own public repository:

`https://github.com/Immrudul/test-repo`

This is based on part of an open source blindfolded Rubik's Cube solving project I worked on previously, except I intentionally introduced a bug and added a pytest suite so that the agent has something real to debug.

---

# Program flow

The basic flow looks like:

```text
GitHub Repository + Debugging Task
                |
                v
        Solari Sandbox
                |
        clone + dependency setup
                |
                v
        reproduce failing tests
                |
                v
      autonomous debugging agent
                |
        controlled tool calls
     /        |        |       \
 read      search     edit     test
 files     files      code
     \        |        |       /
                v
        debugging trajectory
                |
                v
       independent verification
        pytest + git diff
                |
                v
          trace evaluator
                |
                v
     structured training/eval data
```

A normal run looks something like:

```bash
python main.py
   --repo https://github.com/Immrudul/test-repo
   --task "Investigate the failing test, make the smallest justified code change, and verify the fix with pytest."
   --attempts 3
```

The repo and task can also come from environment variables.

For the default example:

* We clone the public test repo into a Solari sandbox
* Install its dependencies
* Run pytest before the model does anything
* Confirm that the bug actually exists
* Give the model a set of controlled debugging tools
* Let the model inspect and modify the repository
* Record every action, its intent, and the resulting observation
* Independently run pytest again after the agent is done
* Capture the final Git diff
* Evaluate the quality of the debugging trajectory
* Save everything into a JSON trace

The important part here is that the LLM does **not** get some built-in code execution environment.

The model only decides what action it wants to take.

For example:

```text
read solve.py
search for edge_order
run pytest
replace this exact snippet
run pytest again
```

Our Python orchestrator receives that action and sends it to the Solari sandbox.

So the actual execution environment is always Solari.

---

# Running without a model API key

Since not everyone trying the project will necessarily have a Gemini/OpenAI/etc API key available, the project also includes a deterministic demo trajectory. (Although I would highly reccomend using some free API key to test it out! so you can see the following pipeline play out in real time!):

```text
sandbox
→ actions
→ observations
→ edits
→ tests
→ patch
→ trace
→ evaluation
```

The demo trace is intentionally more realistic than just:

```text
read bug
→ fix bug
→ pass tests
```

Instead it includes:

* searching for where the failing value is generated
* inspecting the wrong part of the code initially
* reading the test itself
* reproducing the failure
* revisiting relevant code
* forming a wrong hypothesis
* making an unsuccessful temporary edit
* running tests and seeing that it still fails
* reverting that direction
* inspecting the actual solved-edge logic
* identifying the incorrect pair
* making the correct minimal change
* rerunning tests
* capturing the final patch
* successfully finishing

The demo is clearly labeled as synthetic/deterministic and is **not represented as a real model-generated run**.

A real successful model trace along with a batch of multiple runs is also included separately for comparison!

---

# Why am I happy that I got to use Solari?

At the beginning I could have just used subprocesses locally or spun up Docker containers myself but that would kind of miss the entire point of the project. For an autonomous coding agent, we want an environment where it can safely:

* run arbitrary repository commands
* install dependencies
* modify files
* execute tests
* break things
* recover from failures
* experiment without touching the host system

Solari gives us exactly that isolated execution layer. Every repository run gets its own sandbox and the coding agent never directly interacts with my local machine. This also becomes especially useful once we start thinking about repeated debugging experiments. I also thought it was very cool because I've worked on some very similar stuff at one of my previous internships, Plato! Spinning up web based simulations depended heavily on something like Solari where we could have a generalized infrastructure layer for many of the exact problems you were dealing with when building RL environments for browser agents.

---

# Snapshot based debugging experiments

One of the more interesting things added later was using a shared sandbox snapshot as the starting state for multiple debugging attempts.

Instead of doing this:

```text
clone repo
setup repo
run agent

clone repo
setup repo
run agent

clone repo
setup repo
run agent
```

we can do:

```text
clone repo
setup dependencies
reproduce failure
        |
        v
 snapshot failing state
        |
   +----+----+
   |    |    |
   v    v    v
 run1  run2  run3
```

Every debugging trajectory now starts from the exact same:

* repository state
* installed dependencies
* filesystem
* failing test
* environment

which makes comparisons between different attempts much more meaningful.

For example:

```text
Attempt 1
FAILED
21 steps
No edit
Score: 4.4

Attempt 2
SUCCESS
9 steps
Minimal patch
Score: 9.8

Attempt 3
SUCCESS
14 steps
Extra investigation
Score: 8.7
```

This becomes useful not only for evaluating whether an agent solved something, but how it solved it.

---

# Important questions to be asked

## 1. What should actually count as an agent action?

This is basically the equivalent of the granularity question from my previous developer telemetry project. Do we want every token the model generated? Every internal thought? Every shell command? Every read? For this project I decided to keep the trace at a higher and much more useful level. An action is something observable that interacts with the environment:

* `read_file`
* `search_files`
* `run_command`
* `replace_text`
* `write_file`
* `run_tests`
* `finish`

This keeps the data structured and means we don't need to store hidden model chain-of-thought. Instead each action also includes a concise `intent`.

For example:

```json
{
  "intent": "Inspect the solved-edge initialization because the failing order contains unexpected edge labels.",
  "action": {
    "type": "read_file",
    "args": {
      "path": "solve.py",
      "line_start": 100,
      "line_end": 160
    }
  }
}
```

That gives us enough information to understand why the agent chose an action without trying to capture private internal reasoning.

---

## 2. What does a good debugging trajectory actually mean?

A correct patch isn't automatically a good debugging trajectory. Imagine two agents both solve the exact same bug.

Agent A:

```text
read failure
read relevant function
identify incorrect mapping
make one edit
run tests
pass
```

Agent B:

```text
read 20 files
rerun the same test 6 times
make 3 unrelated edits
revert 2
eventually stumble into the right fix
```

Both technically solved the task. But one trajectory is obviously much better training/evaluation data. So I reused the same general rubric from my earlier project:

```text
1. Logical Flow
   Did the agent move from evidence → investigation → edit → verification?

2. Clarity
   What proportion of model actions include concise intents?

3. Efficiency
   Did the agent converge without unnecessary repeated investigation?

4. Evidence
   Did the agent actually inspect code, reproduce failures, and verify the result?

5. Accuracy
   Did the model make a real edit that independently passed the final tests?
```

The difference this time is that most of the evaluator can be deterministic. We already have:

* exact tool calls
* exact observations
* test exit codes
* patch presence
* model vs orchestrator provenance
* edit counts
* read counts
* cached read counts
* first edit step
* tests after edit

so instead of asking another LLM to guess whether the trace was good, we can calculate a large portion of the score from actual execution data. But adding an LLM-as-a-judge like my previous project is not a bad idea at all and is a good mix of deterministic + AI based correctness/grading signals.

---

## 3. How do we know that the agent didn't just claim it fixed the problem?

We just don't trust it LOL. This is probably one of the most important design choices in the project. If the model says something like:

```text
Everything is fixed and tests pass.
```

that doesn't really mean anything by itself. After every run, the orchestrator independently executes:

```bash
python3 -m pytest -q
```

and:

```bash
git diff --
```

These happen outside of the model's control.

The final trace therefore contains:

```json
"final_verification": {
  "tests": {
    "exit_code": 0
  },
  "patch": "..."
}
```

The evaluator considers the run successful based on that result, not based on what the model says in its final response.

---

## 4. How do we stop the agent from wasting all of its context rereading the same code?

This actually became such a real problem during development...

Some weaker / cheaper models would repeatedly request overlapping sections of the same file instead of making progress.

Something like:

```text
read lines 80-160
read lines 100-180
read lines 90-150
read lines 110-170
```

So the orchestrator keeps a history of recent reads. If a request substantially overlaps a recent read, we can return cached evidence instead of spending another real sandbox action. If the model keeps requesting the same thing after already receiving the cached result, the read is eventually blocked and the model is told to either:

* inspect a genuinely different region
* run a test
* form a hypothesis
* make an edit

This turned out to be useful in two ways:

1. reducing unnecessary execution/context
2. creating an interesting measurable signal for debugging-agent efficiency

The evaluator can now literally say:

```text
7 cached reads
1 duplicate investigation
no edit occurred
```

which gives us a pretty clear picture of why a run was bad.

---

## 5. What happens when the model provider fails?

This came up a LOT more than expected because I was mostly using free API tiers during development.

For example Gemini would eventually return a 429 when I exhausted the request quota.

Originally that would just look like:

```text
runtime_error
```

which isn't really accurate.

So provider failures are tracked separately:

```json
{
  "reason": "provider_rate_limit",
  "provider": "gemini",
  "retryable": true
}
```

Even if a provider fails halfway through the run, the system still tries to:

* run final verification
* capture the Git diff
* evaluate whatever trajectory exists
* save the trace

This means failed agent runs are still useful data. #nowastage

---

# Trace

One of the main outputs of the project is a structured debugging trace. A simplified successful trace looks like:

```json
 {                                                                                                                                                                                                                       
    "metadata": {                                                                                                                                                                                                         
      "repo_url": "https://github.com/Immrudul/test-repo",                                                                                                                                                                
      "task": "Investigate the failing test, make the smallest justified code change, and verify the fix with pytest.",                                                                                                   
      "run_kind": "model_agent",                                                                                                                                                                                          
      "provider": "gemini",                                                                                                                                                                                               
      "model": "gemini-3.5-flash-lite"                                                                                                                                                                                    
    },                                                                                                                                                                                                                    
    "baseline": {                                                                                                                                                                                                         
      "tests": {                                                                                                                                                                                                          
        "exit_code": 1                                                                                                                                                                                                    
      }                                                                                                                                                                                                                   
    },                                                                                                                                                                                                                    
    "steps": [                                                                                                                                                                                                            
      {                                                                                                                                                                                                                   
        "step": 1,                                                                                                                                                                                                        
        "actor": "model",                                                                                                                                                                                                 
        "intent": "Search for test_cube_order or edge_order in the repository",                                                                                                                                           
        "action": {                                                                                                                                                                                                       
          "type": "search_files",                                                                                                                                                                                         
          "args": {                                                                                                                                                                                                       
            "query": "edge_order",                                                                                                                                                                                        
            "path": "."                                                                                                                                                                                                   
          }                                                                                                                                                                                                               
        }                                                                                                                                                                                                                 
      },                                                                                                                                                                                                                  
      {                                                                                                                                                                                                                   
        "step": 5,                                                                                                                                                                                                        
        "actor": "model",                                                                                                                                                                                                 
        "intent": "Fix L and F solved check in find_already_solved",                                                                                                                                                      
        "action": {                                                                                                                                                                                                       
          "type": "replace_text",                                                                                                                                                                                         
          "args": {                                                                                                                                                                                                       
            "path": "solve.py"                                                                                                                                                                                            
          }                                                                                                                                                                                                               
        }                                                                                                                                                                                                                 
      }                                                                                                                                                                                                                   
    ],                                                                                                                                                                                                                    
    "final_verification": {                                                                                                                                                                                               
      "tests": {                                                                                                                                                                                                          
        "exit_code": 0                                                                                                                                                                                                    
      },                                                                                                                                                                                                                  
      "patch": "..."                                                                                                                                                                                                      
    },                                                                                                                                                                                                                    
    "evaluation": {                                                                                                                                                                                                       
      "execution": {                                                                                                                                                                                                      
        "verified_success": true                                                                                                                                                                                          
      },                                                                                                                                                                                                                  
      "agent_quality": {                                                                                                                                                                                                  
        "overall": 10.0,                                                                                                                                                                                                  
        "scores": {                                                                                                                                                                                                       
          "logical_flow": { "score": 10 },                                                                                                                                                                                
          "clarity": { "score": 10 },                                                                                                                                                                                     
          "efficiency": { "score": 10 },                                                                                                                                                                                  
          "evidence": { "score": 10 },                                                                                                                                                                                    
          "accuracy": { "score": 10 }                                                                                                                                                                                     
        }                                                                                                                                                                                                                 
      }                                                                                                                                                                                                                   
    },                                                                                                                                                                                                                    
    "success": true                                                                                                                                                                                                       
  }
```

There is also a raw `commands` section containing the actual sandbox commands that produced the observations. This separation matters because something like:

```json
{
  "action": {
    "type": "run_tests"
  }
}
```

is the semantic action requested by the model.

Whereas:

```json
{
  "command": "python3",
  "args": ["-m", "pytest", "-q"]
}
```

is what actually happened inside Solari.

---

# Architecture

## DebugAgent

The `DebugAgent` is basically the brain. It receives:

* the task
* baseline failure
* current debugging state
* available tool schemas
* recent observations

The important restriction is that the model does not get arbitrary host execution. Instead, it chooses from our tool interface. That lets us keep model decision making separate from environment execution.

---

## ActionExecutor

The `ActionExecutor` maps model actions into real sandbox operations.

For example:

```text
read_file
```

becomes a controlled `sed` call.

```text
search_files
```

becomes a repository search.

```text
replace_text
```

executes an exact replacement and asserts that the requested snippet appears exactly once.

```text
run_tests
```

always runs the configured pytest suite.

This keeps mutations small and observable.

---

## Solari Sandbox

Solari is the actual compute environment. The sandbox is responsible for:

* cloning repositories
* installing dependencies
* file inspection
* file editing
* running tests
* command execution
* final verification

This gives each debugging attempt an isolated environment and prevents an autonomous model from modifying the host machine.

---

## TraceLogger

The trace logger records:

* metadata
* baseline tests
* model actions
* action intents
* observations
* raw sandbox commands
* actor provenance
* cached reads
* edits
* final pytest output
* Git diff
* evaluator output
* termination reason

One design detail I care about here is provenance. For example, if the model requests a repeated read and the orchestrator returns cached evidence, the trace records:

```json
{
  "actor": "model",
  "purpose": "cached_read",
  "execution": {
    "actor": "orchestrator",
    "kind": "cached_read"
  }
}
```

So we don't accidentally make it look like Solari executed something that it didn't. The same applies to the deterministic demo.

---

## TraceEvaluator

The evaluator separates two things:

### Execution quality

Did the environment actually show:

* a failing baseline?
* a successful edit?
* a captured patch?
* passing final tests?

### Agent quality

How good was the model trajectory?

The current deterministic metrics include things like:

```text
model step count
model edit count
first edit step
investigations before edit
tests after edit
passing tests after edit
failed model actions
cached model reads
duplicate investigations
intent coverage
```

These feed into:

```text
Logical Flow
Clarity
Efficiency
Evidence
Accuracy
```

A deterministic demo run is never given an autonomous model score because the edit wasn't selected by the model.

---

# Snapshot / multi-trajectory experiments

The project can also run multiple debugging attempts against the same initial failure.

The idea is:

```text
Prepare sandbox
      |
      v
Run baseline tests
      |
      v
Create snapshot
      |
 +----+----+
 |    |    |
 v    v    v
 A    B    C
 |    |    |
 v    v    v
trace trace trace
 \    |    /
      v
    compare
```

This lets us compare debugging trajectories fairly.

Instead of simply asking:

> Did the agent fix the bug?

we can ask:

> Which attempt found the best verified fix with the strongest debugging trajectory?

This also starts getting pretty close to automatically creating preference data.

For example:

```json
{
  "chosen": "attempt_2",
  "rejected": "attempt_1",
  "chosen_score": 9.8,
  "rejected_score": 4.4
}
```

Both attempts started from the exact same environment, so the comparison is much cleaner.

---

# Example results

One real successful autonomous run looked like:

```text
Baseline:
1 failed

Agent:
9 model actions
1 code edit
first edit on step 7
1 passing test after edit

Final verification:
1 passed
patch captured

Evaluation:
Logical Flow   10/10
Clarity        10/10
Efficiency      9/10
Evidence       10/10
Accuracy       10/10

Overall:
9.8/10
```

Another real run against the exact same bug failed very differently:

```text
21 model actions
19 investigation actions
7 cached reads
0 edits

provider rate limit reached

Final verification:
1 failed

Overall:
4.4/10
```

That difference is actually one of the more interesting outputs of the project. The failed trajectory isn't useless. It's an example of an agent that repeatedly gathered evidence but failed to turn that evidence into an actionable diagnosis.

---

# Improvements or things to improve on after our implemented MVP

Of course this is still a pretty scrappy project and there are a million directions this could go. At some point though I think adding more features starts becoming diminishing returns for a project, so I stopped once the core experimentation loop felt complete. Some obvious next steps would be:

* support repository-specific setup/test commands instead of assuming Python + `requirements.txt` + pytest

  * this could make it support npm, cargo, go test, etc.

* run experiments across multiple models

  * same snapshot
  * same task
  * Gemini vs OpenAI vs Groq vs local models
  * compare success rate, steps, efficiency, and patch quality

* automatically generate benchmark suites

  * N repositories
  * M models
  * K attempts per task
  * output leaderboard + aggregate metrics

* build a UI for comparing trajectories

  * action timeline
  * intents
  * file reads
  * test outputs
  * code diffs
  * evaluator scores

* convert multiple trajectories from the same task into explicit preference datasets

  * successful vs failed
  * efficient vs inefficient
  * minimal patch vs unnecessarily large patch

* add more semantic evaluation for intent quality

  * right now clarity measures whether concise intents are present, not whether those intents are actually insightful

* track token usage / inference cost

  * especially useful when comparing debugging efficiency between models

* run attempts concurrently from the same snapshot

  * right now provider quota limits make sequential execution easier during development

The bigger version of this project would basically become a small benchmark and trajectory generation platform for autonomous coding agents. But for now I think the current version does a pretty good job of showing the full loop:

```text
real bug
→ autonomous investigation
→ sandbox execution
→ code changes
→ deterministic verification
→ structured trajectory
→ measurable debugging quality
```

which was really the part I wanted to explore.

Honestly I know that was a lot but thank you for reading all that! Please feel free to clone the project, and of course feel free to reach out about anything as well!

My sincere thank yous,
Mrudul

suresh.mrudul@gmail.com
mrudul.suresh@uwaterloo.ca
