
import sys

def parse_kosarak_to_sparse_arff(file_path):
    with open(file_path, 'r') as file:
        transactions = [line.strip().split() for line in file]

    unique_items = set(item for transaction in transactions for item in transaction)
    
    sorted_items = sorted(unique_items)

    item_index = {item: idx for idx, item in enumerate(sorted_items)}

    arff_content = []
    arff_content.append("@relation kosarak")

    for item in sorted_items:
        arff_content.append(f"@attribute item_{item} {{0, 1}}")
    
    arff_content.append("@data")
    for transaction in transactions:
        indices = [str(item_index[item]) for item in transaction]
        arff_content.append("{" + ",".join(f"{idx} 1" for idx in indices) + "}")
    
    return "\n".join(arff_content)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_kosarak_data>")
    else:
        file_path = sys.argv[1]
        arff_output = parse_kosarak_to_sparse_arff(file_path)
        print(arff_output)
