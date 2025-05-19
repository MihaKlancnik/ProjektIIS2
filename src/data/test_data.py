import sys
import pandas as pd
import os
from evidently import Report
from evidently.presets.dataset_stats import DataSummaryPreset
from evidently.presets.drift import DataDriftPreset

def test_crypto_data(crypto_name):
    """
    Tests data quality and drift for a given cryptocurrency dataset using Evidently.

    Args:
        crypto_name (str): The name of the cryptocurrency (e.g., 'bitcoin', 'ethereum', 'solana').

    Returns:
        int: 0 if tests passed, 1 if tests failed.
    """
    current_data_path = f"data/preprocessed/price/{crypto_name}.csv"
    reference_data_path = f"data/reference/price/{crypto_name}.csv"
    report_output_path = f"reports/{crypto_name}_data_testing_report.html"

    print(f"\n--- Testing data for {crypto_name.capitalize()} ---")

    # Create necessary directories if they don't exist
    os.makedirs(os.path.dirname(reference_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)

    # Load the current data
    try:
        current = pd.read_csv(current_data_path)
        print(f"Loaded current data from {current_data_path}")
        if current.empty:
            print(f"Warning: Current data file for {crypto_name} is empty.")
            return 1 # Fail if current data is empty
    except FileNotFoundError:
        print(f"Error: Current data file not found at {current_data_path}")
        return 1
    except Exception as e:
        print(f"Error loading current data for {crypto_name}: {e}")
        return 1

    # Variable to hold the reference DataFrame, initialized to None
    reference = None

    # Check if reference data exists and create if necessary
    if not os.path.exists(reference_data_path):
        print(f"Reference file not found for {crypto_name}. Copying from current data to {reference_data_path}.")
        try:
            # Ensure current data is not empty before creating reference
            if not current.empty:
                 current.to_csv(reference_data_path, index=False)
                 print(f"Created initial reference data at {reference_data_path}")
                 # --- FIX: Load the newly created reference data ---
                 try:
                     reference = pd.read_csv(reference_data_path)
                     print(f"Loaded newly created reference data from {reference_data_path}")
                 except Exception as e:
                     print(f"Error loading newly created reference data for {crypto_name}: {e}")
                     # If we just saved it but can't load it, something is wrong.
                     return 1
                 # --- END FIX ---
            else:
                 print(f"Cannot create reference data for {crypto_name} as current data is empty.")
                 return 1 # Fail if we can't even create an initial reference
        except Exception as e:
            print(f"Error creating reference file for {crypto_name}: {e}")
            return 1
    else:
        # Load the reference data if it exists
        try:
            reference = pd.read_csv(reference_data_path)
            print(f"Loaded reference data from {reference_data_path}")
            if reference.empty:
                 print(f"Warning: Reference data file for {crypto_name} is empty.")
                 # Decide if empty reference data should be a failure. Yes, we can't compare.
                 return 1
        except Exception as e:
            print(f"Error loading reference data for {crypto_name}: {e}")
            return 1

    # --- Ensure reference was successfully loaded before proceeding ---
    if reference is None:
         print(f"Error: Reference data could not be loaded or created for {crypto_name}.")
         return 1

    # Ensure both dataframes have the same columns for comparison
    # In this case, 'timestamp' and 'price' should be consistent, but a general check is good.
    if not reference.columns.equals(current.columns):
        print(f"Error: Reference and current data columns do not match for {crypto_name}.")
        print(f"Reference columns: {list(reference.columns)}")
        print(f"Current columns: {list(current.columns)}")
        return 1


    # Define and run the Evidently report
    # We include DataSummaryPreset and DataDriftPreset as in the example
    report = Report([
        DataSummaryPreset(),
        DataDriftPreset(),
    ],
    include_tests=True # Generate tests based on the presets
    )

    try:
        result = report.run(reference_data=reference, current_data=current)
        print("Evidently report generated.")
    except Exception as e:
        print(f"Error generating Evidently report for {crypto_name}: {e}")
        return 1

    # Save the report to an HTML file
    try:
        result.save_html(report_output_path)
        print(f"Report saved to {report_output_path}")
    except Exception as e:
        print(f"Error saving report for {crypto_name}: {e}")
        # Continue to check tests even if saving fails, but log the error.
        pass


    # Check if the report contains any tests and if all tests passed
    all_tests_passed = True
    tests_found = False
    try:
        result_dict = result.dict()
        if "tests" in result_dict and result_dict["tests"]:
            tests_found = True
            for test in result_dict["tests"]:
                # Check for 'status' and handle potential missing keys defensively
                status = test.get("status")
                test_name = test.get("name", "Unknown Test")
                if status != "SUCCESS":
                    all_tests_passed = False
                    print(f"  Test Failed: {test_name} (Status: {status})")
        elif "tests" in result_dict and not result_dict["tests"]:
             print("Evidently report generated, but no tests were found in the result.")
             # Depending on requirements, you might want to fail if no tests run.
             # For now, we'll treat it as passed if no tests were configured/found.
             all_tests_passed = True # Or False if you require tests to be present
        else:
             print("No 'tests' section found in the Evidently report result dictionary.")
             # Again, decide strictness. Treating as passed for now.
             all_tests_passed = True # Or False

    except Exception as e:
        print(f"Error processing test results for {crypto_name}: {e}")
        all_tests_passed = False # Assume failure if results can't be processed

    # Output test result and handle reference data
    if not all_tests_passed:
        print(f"Data tests failed for {crypto_name.capitalize()}.")
        return 1
    else:
        print(f"Data tests passed for {crypto_name.capitalize()}.")
        # Replace the reference data with the current data if tests passed and reference existed
        # We check if os.path.exists(reference_data_path) again to be safe,
        # although if reference was loaded successfully, it should exist.
        if os.path.exists(reference_data_path) and reference is not None:
            try:
                # Remove old reference and save current as new reference
                os.remove(reference_data_path)
                current.to_csv(reference_data_path, index=False)
                print(f"Reference data updated for {crypto_name.capitalize()}.")
            except Exception as e:
                print(f"Error updating reference data for {crypto_name}: {e}")
                # This is a critical step; failing here might mean the next run compares
                # against outdated data. Consider exiting with an error here too if strict.
                # For now, we just print the error but let the overall script continue
                # as the data *tests* themselves passed.
                pass
        elif tests_found: # If reference didn't exist and tests ran (unlikely scenario with fix)
             print(f"Reference data did not exist for {crypto_name}, no update needed after passing tests.")

        return 0

# Main execution block
if __name__ == "__main__":
    # Define the list of cryptocurrencies to test
    cryptos_to_test = ["bitcoin", "ethereum", "solana"]
    overall_status = 0 # 0 for success, 1 for failure

    # Run tests for each cryptocurrency
    for crypto in cryptos_to_test:
        status = test_crypto_data(crypto)
        if status != 0:
            overall_status = 1 # If any crypto test fails, the overall status is failure

    print("\n--- Overall Data Testing Summary ---")
    if overall_status == 0:
        print("All cryptocurrency data tests passed.")
    else:
        print("One or more cryptocurrency data tests failed.")

    # Exit the script with the overall status code
    sys.exit(overall_status)