# AVX Instruction Set Demo

This project demonstrates the usage of AVX, AVX2, and AVX512 instruction sets in C++. It provides a framework for testing and verifying SIMD (Single Instruction, Multiple Data) operations against their SISD (Single Instruction, Single Data) scalar counterparts.

## Features

The project covers a wide range of vector operations across different AVX extensions.

### Supported Functionality Matrix

| Category | Class Type | Operations | Supported Types |
|----------|------------|------------|-----------------|
| **AVX**    | arithmetic | add, sub, mul, div | int, float, double |
| **AVX2**   | arithmetic | add, sub, mul, div | int, float |
|          | bitwise    | and, or, xor, andnot | int, float, double |
|          | blend      | blend | int, float, double |
|          | cmp        | cmp_eq, cmp_gt, cmp_lt, ... | int, float, double |
|          | fma        | fmadd, fmsub, fnmadd, fnmsub | float, double |
|          | minmax     | min, max | int, float, double |
|          | permute    | shuffle, unpacklo, unpackhi, permutevar | int |
|          | shift      | sll, srl, sra | int |
| **AVX512** | arithmetic | add, sub, mul, div | int, float, double |
|          | bitwise    | and, or, xor, andnot | int, float, double |
|          | blend      | blend | int, float, double |
|          | cmp        | cmp_eq, cmp_gt, cmp_lt, ... | int, float, double |
|          | compress   | compress | int, float, double |
|          | fma        | fmadd, fmsub, fnmadd, fnmsub | float, double |
|          | mask       | mask_add, mask_sub, ... | int, float, double |
|          | permute    | permute, shuffle, unpack | int, float, double |
|          | reduce     | reduce_add, reduce_mul, ... | int, float, double |
|          | shift      | sll, srl, sra, rol, ror | int |
|          | vnni       | vnni (dpbusd) | uint8_t, int8_t, int16_t |

### Calculation Logic

The framework uses a dual-path verification approach:
1.  **AVX Path**: Executes the operation using intrinsic functions (e.g., `_mm256_add_ps`).
2.  **SISD Path**: Executes the same logic using scalar C++ code.
3.  **Verification**: Compares the results of both paths to ensure correctness.

## Usage

### Compilation

Ensure you have a C++17 compliant compiler (like g++) that supports AVX512.

```bash
make
```

### Running Tests

The `avx` executable supports flexible filtering to run specific subsets of tests.

```bash
./avx [options]
```

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--avx<version>` | Filter by AVX category. | `--avx2` (Runs all AVX2 tests) |
| `--class=<name>` | Filter by class type. | `--class=cmp` (Runs comparison tests) |
| `--op=<name>` | Filter by operation name (partial match). | `--op=add` (Runs operations containing "add") |
| `--type=<sig>` | Filter by type signature (partial match). | `--type=float` (Runs tests involving float) |
| `--help` | Show help message. | `./avx --help` |

### Hierarchical Help

You can combine `--help` with other filters to see available options at that level.

*   **List all categories:**
    ```bash
    ./avx --help
    ```
*   **List classes in a category:**
    ```bash
    ./avx --avx2 --help
    ```
*   **List operations in a class:**
    ```bash
    ./avx --avx2 --class=permute --help
    ```

## Example Outputs

### 1. AVX512 VNNI (Vector Neural Network Instructions)

```bash
$ ./avx --avx512 --class=vnni
avx512:vnni:vnni<uint8_t[64], int8_t[64], int[16] -> int[16]>
a = 192 179 25 169 243 ...
b = -45 0 -70 -99 20 ...
c = 52 -75 -46 3 78 ...
avx_e = -27069 -6349 -16269 ...
sisde = -27069 -6349 -16269 ...
```

### 2. AVX2 Permute Operations

```bash
$ ./avx --avx2 --class=permute
avx2:permute:shuffle_0123<int[8] -> int[8]>
a = -100 -59 -80 30 3 25 51 90 
avx_c = 30 -80 -59 -100 90 51 25 3 
sisdc = 30 -80 -59 -100 90 51 25 3 
...
```

### 3. Filtering by Operation Name

```bash
$ ./avx --op=min
avx2:minmax:min<int[8] -> int[8]>
...
avx2:minmax:min<float[8] -> float[8]>
...
```
