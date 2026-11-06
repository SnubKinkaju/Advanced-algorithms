# Contributing

## Adding a method

1. Create one file per method in `src/setcover/methods/`, for example `greedy.py`.
2. The file exposes one function:

   ```python
   def solve(inst: Instance, time_limit: float, rng: np.random.Generator) -> Result:
   ```

3. Register it in `src/setcover/registry.py`.
4. Add your tests in `tests/test_<method>.py` and run `pytest` before pushing.

`src/setcover/methods/trivial.py` is a minimal template that follows all the rules below.

## Rules for every method

- Indices start at 0: elements are `0..n-1`, subsets are `0..m-1`.
- Instance files are read only through `setcover.io.read_instance`.
- Never modify the instance. To reduce it, build a new one with `make_instance`.
- Randomness comes only from the `rng` argument. Do not call `random.seed` or `np.random.seed`.
- Compute `deadline = time.perf_counter() + time_limit` at the start and check it in the main loop. When it is reached, return the best cover found so far with `timed_out=True`.
- `status` is `"optimal"` only if optimality is proved, `"feasible"` for a valid cover without proof, `"none"` if no cover was found.
- Set `lower_bound` when the method proves one.
- Method-specific parameters are keyword arguments with default values, for example `def solve(inst, time_limit, rng, t0=10.0)`.
- Do not measure the running time inside `solve`: the experiment runner does it.
- Never read `data/reference.csv`.
- Do not print inside `solve`.

## Workflow

- One branch per issue (`feat/<short-name>`), then a pull request into `main`, reviewed by another member.
- Write `Closes #<issue number>` in the pull request description.
- Fill the "Time spent (h)" field of the issue.
