import sys
import great_expectations as gx

if len(sys.argv) != 2:
    print("Usage: python run_checkpoint.py <checkpoint_name>")
    sys.exit(1)

checkpoint_name = sys.argv[1]

context = gx.get_context()

try:
    checkpoint = context.get_checkpoint(checkpoint_name)
except Exception as e:
    print(f"Checkpoint '{checkpoint_name}' not found. Error: {e}")
    sys.exit(1)

print("BatchRequest configuration:")
print(checkpoint.config.batch_request)

run_id = f"{checkpoint_name}_run"
try:
    checkpoint_result = checkpoint.run(run_id=run_id)
except Exception as e:
    print(f"Error during checkpoint run: {e}")
    sys.exit(1)

context.build_data_docs()

if checkpoint_result["success"]:
    print(f"Validation for {checkpoint_name} passed!")
    sys.exit(0)
else:
    print(f"Validation for {checkpoint_name} failed!")
    sys.exit(1)