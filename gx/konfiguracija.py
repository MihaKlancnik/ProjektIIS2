import great_expectations as gx

context = gx.get_context()

# Create a new Datasource
datasource_name = "crypto_data"
datasource = context.sources.add_pandas_filesystem(
    name=datasource_name,
    base_directory=r"data\preprocessed"
)

# Define CSV files and corresponding asset names and expectation suites
assets_info = [
    {
        "path": r"price\\bitcoin.csv",
        "name": "bitcoin_data",
        "suite": "bitcoin_suite",
        "checkpoint": "bitcoin_checkpoint"
    },
    {
        "path": r"price\\ethereum.csv",
        "name": "ethereum_data",
        "suite": "ethereum_suite",
        "checkpoint": "ethereum_checkpoint"
    },
    {
        "path": r"price\\solana.csv",
        "name": "solana_data",
        "suite": "solana_suite",
        "checkpoint": "solana_checkpoint"
    },
    {
        "path": r"fear_greed\\fear_greed_index.csv",
        "name": "fear_greed_data",
        "suite": "fear_greed_suite",
        "checkpoint": "fear_greed_checkpoint"
    }
]

for asset_info in assets_info:
    # Add data asset
    asset = datasource.add_csv_asset(
        name=asset_info["name"],
        batching_regex=asset_info["path"]
    )

    # Create or update expectation suite
    expectation_suite = context.add_or_update_expectation_suite(
        expectation_suite_name=asset_info["suite"]
    )

    # Generate expectations suite using the onboarding assistant
    asset_obj = context.get_datasource(datasource_name).get_asset(asset_info["name"])
    batch_request = asset_obj.build_batch_request()
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=asset_info["suite"]
    )

    data_assistant_result = context.assistants.onboarding.run(
        validator=validator
    )

    # Save expectation suite
    expectation_suite = data_assistant_result.get_expectation_suite()
    context.save_expectation_suite(
        expectation_suite=expectation_suite,
        expectation_suite_name=asset_info["suite"]
    )

    # Create checkpoint
    checkpoint = context.add_or_update_checkpoint(
        name=asset_info["checkpoint"],
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": asset_info["suite"]
            }
        ],
    )

# Build data docs
site = context.build_data_docs()
context.open_data_docs()
