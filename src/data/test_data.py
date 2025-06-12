import sys
import pandas as pd
import os
from evidently import Report
from evidently.presets.dataset_stats import DataSummaryPreset
from evidently.presets.drift import DataDriftPreset
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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


    os.makedirs(os.path.dirname(reference_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)

    try:
        current = pd.read_csv(current_data_path)
        print(f"Loaded current data from {current_data_path}")
        if current.empty:
            print(f"Warning: Current data file for {crypto_name} is empty.")
            return 1
    except FileNotFoundError:
        print(f"Error: Current data file not found at {current_data_path}")
        return 1
    except Exception as e:
        print(f"Error loading current data for {crypto_name}: {e}")
        return 1


    reference = None


    if not os.path.exists(reference_data_path):
        print(f"Reference file not found for {crypto_name}. Copying from current data to {reference_data_path}.")
        try:

            if not current.empty:
                 current.to_csv(reference_data_path, index=False)
                 print(f"Created initial reference data at {reference_data_path}")
                 try:
                     reference = pd.read_csv(reference_data_path)
                     print(f"Loaded newly created reference data from {reference_data_path}")
                 except Exception as e:
                     print(f"Error loading newly created reference data for {crypto_name}: {e}")
                     return 1
            else:
                 print(f"Cannot create reference data for {crypto_name} as current data is empty.")
                 return 1 
        except Exception as e:
            print(f"Error creating reference file for {crypto_name}: {e}")
            return 1
    else:

        try:
            reference = pd.read_csv(reference_data_path)
            print(f"Loaded reference data from {reference_data_path}")
            if reference.empty:
                 print(f"Warning: Reference data file for {crypto_name} is empty.")
                 return 1
        except Exception as e:
            print(f"Error loading reference data for {crypto_name}: {e}")
            return 1

    if reference is None:
         print(f"Error: Reference data could not be loaded or created for {crypto_name}.")
         return 1


    if not reference.columns.equals(current.columns):
        print(f"Error: Reference and current data columns do not match for {crypto_name}.")
        print(f"Reference columns: {list(reference.columns)}")
        print(f"Current columns: {list(current.columns)}")
        return 1



    report = Report([
        DataSummaryPreset(),
        DataDriftPreset(),
    ],
    include_tests=True
    )

    try:
        result = report.run(reference_data=reference, current_data=current)
        print("Evidently report generated.")
    except Exception as e:
        print(f"Error generating Evidently report for {crypto_name}: {e}")
        return 1


    try:
        result.save_html(report_output_path)
        print(f"Report saved to {report_output_path}")
    except Exception as e:
        print(f"Error saving report for {crypto_name}: {e}")
        pass



    all_tests_passed = True
    tests_found = False
    try:
        result_dict = result.dict()
        if "tests" in result_dict and result_dict["tests"]:
            tests_found = True
            for test in result_dict["tests"]:
                status = test.get("status")
                test_name = test.get("name", "Unknown Test")
                if status != "SUCCESS":
                    all_tests_passed = False
                    print(f"  Test Failed: {test_name} (Status: {status})")
        elif "tests" in result_dict and not result_dict["tests"]:
             print("Evidently report generated, but no tests were found in the result.")

             all_tests_passed = True 
        else:
             print("No 'tests' section found in the Evidently report result dictionary.")

             all_tests_passed = True 

    except Exception as e:
        print(f"Error processing test results for {crypto_name}: {e}")
        all_tests_passed = False


    if not all_tests_passed:
        print(f"Data tests failed for {crypto_name.capitalize()}.")
        return 1
    else:
        print(f"Data tests passed for {crypto_name.capitalize()}.")

        if os.path.exists(reference_data_path) and reference is not None:
            try:

                os.remove(reference_data_path)
                current.to_csv(reference_data_path, index=False)
                print(f"Reference data updated for {crypto_name.capitalize()}.")
            except Exception as e:
                print(f"Error updating reference data for {crypto_name}: {e}")

                pass
        elif tests_found:
             print(f"Reference data did not exist for {crypto_name}, no update needed after passing tests.")

        return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test crypto data or update reference data.")
    parser.add_argument('--update-reference', action='store_true', help='Update reference data with current data for all cryptos.')
    args = parser.parse_args()

    cryptos_to_test = ["bitcoin", "ethereum", "solana"]
    overall_status = 0 

    if args.update_reference:
        for crypto in cryptos_to_test:
            current_data_path = f"data/preprocessed/price/{crypto}.csv"
            reference_data_path = f"data/reference/price/{crypto}.csv"
            try:
                current = pd.read_csv(current_data_path)
                current.to_csv(reference_data_path, index=False)
                print(f"Reference data updated for {crypto.capitalize()} (forced update).")
            except Exception as e:
                print(f"Error updating reference data for {crypto}: {e}")
                overall_status = 1
        sys.exit(overall_status)


    for crypto in cryptos_to_test:
        status = test_crypto_data(crypto)
        if status != 0:
            overall_status = 1 

    print("\n--- Overall Data Testing Summary ---")
    if overall_status == 0:
        print("All cryptocurrency data tests passed.")
    else:
        print("One or more cryptocurrency data tests failed.")

    sys.exit(overall_status) # to mors pol spremenit, zdj mas tk da ti dela
    #sys.exit(overall_status)

    #VSE SNOVI KO SO NA ESTUDIJU RAZN NADZOROVANJA NEBO
    #ZAPRT TIP VPRASANJ
    #MULTIBLE CHOCE
    #SAM OBKROZEVALI BOMO
    #ustni zagovori
    #petek 13 junij = zagovor projektnih nalog + ustni zagovor
    #7.julij je ustni ce mors 3. na izpit
