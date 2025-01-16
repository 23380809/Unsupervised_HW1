# Problem 2 A - Python program to console log ARFF file from Kosarak dataset
import sys

def parse_kosarak_to_sparse_arff(file_path):
    with open(file_path, 'r') as file:
        transactions = [line.strip().split() for line in file]

    sorted_items = sorted(set(item for transaction in transactions for item in transaction.strip().split()), key=lambda x: int(x))

    item_index = {item: i for i, item in enumerate(sorted_items)}

    content = ["@relation kosarak"]

    for item in sorted_items:
        content.append(f"@attribute item_{item} {{0, 1}}")
    
    content.append("@data")
    for transaction in transactions:
        indices = [str(item_index[item]) for item in transaction]
        content.append("{" + ",".join(f"{i} 1" for i in indices) + "}")
    
    return "\n".join(content)


file_path = sys.argv[1]
arff_output = parse_kosarak_to_sparse_arff(file_path)
print(arff_output)
