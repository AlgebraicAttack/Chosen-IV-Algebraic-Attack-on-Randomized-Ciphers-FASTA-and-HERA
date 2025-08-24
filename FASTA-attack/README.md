
# FASTA Attack Script

This project contains a SageMath script (`fasta-attack.sage`, `construct_fasta_linear_layer.sage`) that performs a FASTA-based cryptographic attack. The script calculates certain polynomials and uses Gaussian elimination over finite fields (GF(2)) to solve linear systems.


## Parameters

The script requires three parameters:

1. **b**: The block size (can be `3` ).
2. **s**: The size of each block (can be `5` or `7`).
3. **r**: The number of rounds (`3`).

You will need to provide values for these parameters when executing the script.

## Usage

### 1. Run the Script

You can run the script from the command line as follows:

```bash
sage fasta-attack.sage 3 5 3
```

This will run the `fasta-attack.sage` script with the following parameters:

* `b = 3` (block size)
* `s = 5` (block size)
* `r = 3` (rounds)

You can change the values of `b` and `s` as required. The `r` parameter is fixed to `3` in this example.

### Example Commands:

* For `b = 3`, `s = 5`, `r = 3`:

  ```bash
  sage fasta-attack.sage 3 5 3
  ```





