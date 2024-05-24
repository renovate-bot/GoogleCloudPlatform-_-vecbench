# vecbench

## Install

As of now vecbench uses Python v3.8. Newer Python versions are currently unsupported.

Using https:
`git clone https://github.com/google/vecbench.git`

Using ssh:
`git clone git@github.com:google/vecbench.git`

Install a virtual environment:

`python3 -m venv venv`

Activate it:

`source venv/bin/activate`

Older Pip releases have an optimistic dependency resolver which might end up installing incompatible library versions and not bothering to find a working version combination. So to make sure you get a recent Pip release with an actual dependency resolver, update it:

`pip install -U pip`

Then install dependencies:

`pip install -r requirements.txt`

For using the [find_recall.sh](./vecbench/find_recall.sh) script you need the bc package. Install it using:

`sudo apt-get install bc`

## Configuration

You will need to configure a DB password if your data store requires one:
- VECBENCH_DB_PASSWORD

Configure the 3 config files needed to run a benchmark:

1. Database config: [DB Config](./vecbench/config/db/alloydb-omni.yaml)
2. Dataset config:  [Dataset Config](./vecbench/config/dataset/glove_100_angular.yaml)
3. Database config: [Benchmark Config](./vecbench/config/benchmark/simple_ann_l2.yaml)

Specify the env variable which loader (local host or ray) to us:
- LOADER=MPLoader for local runs
- LOADER=RAYLoader for [Ray](https://www.ray.io/) runs

Once you have configured the 3 components of vector benchmarking, you can run vecbench:

`python3 vecbench.py --loader $LOADER --db_config  $DB_CONFIG --dataset_config  $DATASET_CONFIG --benchmark_config  $BENCHMARK_CONFIG`

## Making HDF5 files
Added an option to generate HD5F files from binary files.

`python3 vecbench.py --make_hdf5 binary downloads/binfiles/query.public.10K.u8bin downloads/query.hdf5 query`

## Existing Datasets

[To be created]

## Making contributions

When adding a new dependency to `requirements.txt`, please pin its version with `==`, (e.g. `mydependency==1.0.1`). Make sure to test that installing `requirements.txt` with the new dependency finishes without any version resolution errors with a recent PIP version (to update your pip version in your venv refer to the [install section](#install)). It shouldn't be necessary to pin the versions of your dependency's dependencies; please avoid it unless you encounter some breakage that can be solved by pinning it.
