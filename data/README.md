# Dataset intake and release

## Source checks

The expected pin in `plan.md` refers to the original 21-column Telco sample supplied by IBM. The SHA-256 is `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`. The source should contain 7,043 rows and 7,043 distinct nonempty IDs, 1,869 `Yes` labels, and 11 whitespace-only `TotalCharges` values on exactly the 11 tenure-zero rows. Record the custodian's measured values and approved attribution and redistribution terms in this file. [IBM source repository](https://github.com/IBM/telco-customer-churn-on-icp4d)

## Custodian intake

In the isolated environment, resolve the IBM commit for the input and record the commit, permanent raw URL, attribution, and applicable redistribution terms above. Set `SOURCE_CSV` to the downloaded file, `PRIVATE_DIR` to a restricted directory outside the checkout and release, and `RELEASE_DIR` to the training release staging directory. Run with the pinned environment active:

```sh
python -m src.split_data --source "$SOURCE_CSV" --private-dir "$PRIVATE_DIR" --release-dir "$RELEASE_DIR"
```

In PowerShell the equivalent is:

```powershell
python -m src.split_data --source "$env:SOURCE_CSV" --private-dir "$env:PRIVATE_DIR" --release-dir "$env:RELEASE_DIR"
```

Record the emitted aggregate checks in this file. The intake produces `PRIVATE_DIR/holdout.csv` with 1,409 rows, and `RELEASE_DIR/train.csv` with 5,634 rows. Both have the `row_id` join key, 19 original predictors, and binary `Churn` (`Yes` = 1, `No` = 0). The release also contains `splits.csv` with `row_id,split` and the values `train` and `holdout`. The source and holdout stay in restricted custodian storage.

Publish the exact `train.csv` bytes as an immutable version. Use authorized private project storage when redistribution terms are unclear. Set `ASSET_VERSION` to the published version and `TRAIN_URL` to the stable HTTPS download location. Keep credentials and temporary signed tokens in the storage session. From the project root:

```sh
python -m src.train_data write-manifest --release-dir "$RELEASE_DIR" --asset-version "$ASSET_VERSION" --download-url "$TRAIN_URL"
python -m src.train_data verify --data-dir "$RELEASE_DIR"
```

In PowerShell:

```powershell
python -m src.train_data write-manifest --release-dir "$env:RELEASE_DIR" --asset-version "$env:ASSET_VERSION" --download-url "$env:TRAIN_URL"
python -m src.train_data verify --data-dir "$env:RELEASE_DIR"
```

Commit `data/train_manifest.json` and `data/splits.csv` from the release directory along with the measured source record in this README. The training manifest records `asset_version`, `download_url`, `byte_count`, `row_count`, `raw_source_sha256`, and `train_sha256`. Include the exact authenticated download command in the developer workflow if the storage provider requires one.

## Release status

Production intake and source measurements are pending with Mohammed Altuwalah, with Majed Alsehli as backup. Developers can verify the foundation with the [fixture checks](../README.md#setup-and-fixture-checks) until those release files arrive.
