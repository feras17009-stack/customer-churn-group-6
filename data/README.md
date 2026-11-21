# Dataset intake and release

## Source checks

IBM supplies the original 21-column Telco sample. The pinned file comes from commit `1fd6fd70906479be1712e64cfc0ea89f58466629` and has SHA-256 `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`. The source pin requires 7,043 distinct customer IDs, 1,869 `Yes` labels, and 11 whitespace-only `TotalCharges` values on tenure-zero rows. IBM publishes the source repository with an Apache 2.0 license. The project keeps the original and holdout in local custodian storage. [Pinned IBM CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv) · [IBM repository license](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/LICENSE)

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

The intake verified 7,043 unique records, 1,869 positive labels and 11 blank total charges, all at zero tenure. It produced 5,634 training rows and 1,409 holdout rows. The training snapshot `ibm-telco-v1` uses a local-only manifest. Its SHA-256 is `03ab0632c9b240ebcbe2eae492ea81a934c5b4ab1c1d078b0f32044893e13865`.

To regenerate the local-only release, run the split command above with separate empty staging and holdout directories. For an initial split, record the manifest with `python -m src.train_data write-manifest --release-dir "$env:RELEASE_DIR" --asset-version ibm-telco-v1 --local-only`. On a checkout that already contains the reviewed manifest and splits, copy the regenerated `train.csv` alone into `data/` and verify it with `python -m src.train_data verify`.
