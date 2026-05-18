import os
import sys

# Make sure tools.py is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools import run_python, search_documents, validate_yaml


def banner(text: str):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def main():
    # --- Test 1: search_documents ---
    banner("TEST 1: search_documents")
    query = "how do I roll back a deployment to the previous revision"
    print(f"Query: {query}\n")
    result = search_documents(query=query, top_k=3)
    print(result)

    # --- Test 2: run_python ---
    banner("TEST 2: run_python")
    code = (
        "import math\n"
        "pods = 12\n"
        "replicas_per_node = 3\n"
        "nodes_needed = math.ceil(pods / replicas_per_node)\n"
        "print(f'Need {nodes_needed} nodes to host {pods} pods at {replicas_per_node} replicas/node')"
    )
    print(f"Code:\n{code}\n")
    result = run_python(code=code)
    print(result)

    # --- Test 3a: validate_yaml (valid input) ---
    banner("TEST 3a: validate_yaml (valid input)")
    valid_yaml = """apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: nginx
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
"""
    print(f"YAML:\n{valid_yaml}")
    result = validate_yaml(yaml_text=valid_yaml)
    print(result)

    # --- Test 3b: validate_yaml (invalid: missing kind) ---
    banner("TEST 3b: validate_yaml (missing 'kind')")
    bad_yaml = """apiVersion: v1
metadata:
  name: my-service
spec:
  selector:
    app: nginx
"""
    print(f"YAML:\n{bad_yaml}")
    result = validate_yaml(yaml_text=bad_yaml)
    print(result)

    # --- Test 3c: validate_yaml (syntax error) ---
    banner("TEST 3c: validate_yaml (broken syntax)")
    broken_yaml = """apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector
    app: nginx
"""
    print(f"YAML:\n{broken_yaml}")
    result = validate_yaml(yaml_text=broken_yaml)
    print(result)

    banner("All tests completed.")


if __name__ == "__main__":
    main()
