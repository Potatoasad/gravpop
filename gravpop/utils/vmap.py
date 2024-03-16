import jax
import jax.numpy as jnp
import os

def is_in_jupyter():
    return "JUPYTER_RUNTIME_DIR" in os.environ


def chunked_vmap(func, in_axes, chunk=100, progress_note=None):
    # Pre-compile the vmap function
    vmap_func = jax.vmap(lambda p: func(p), in_axes)

    from tqdm.auto import tqdm

    # Define a function that applies vmap to input chunks
    def chunk_vmap(params):
        # Get the keys of the input dictionary
        keys = list(params.keys())
        # Get the size of the first key (assuming all keys have the same size)
        size = len(params[keys[0]])

        # Initialize a list to store the results of vmap for each chunk
        results = []

        progress_title = f"{progress_note}" if progress_note else None

        pbar = tqdm(total=size, desc=progress_title)

        # Iterate over the chunks
        for start_idx in range(0, size, chunk):
            end_idx = min(start_idx + chunk, size)

            # Slice the input dictionary to get the chunk
            chunk_params = {key: value[start_idx:end_idx] for key, value in params.items()}

            # Apply vmap to the function with the chunked input
            result = vmap_func(chunk_params)
            results.append(result)

            pbar.update(end_idx - start_idx)

        pbar.close()

        # Concatenate the results along the first axis
        if len(results) == 1:
            return results[0]
        else:
            return jnp.concatenate(results, axis=0)

    # Return the function that applies vmap to input chunks
    return chunk_vmap
