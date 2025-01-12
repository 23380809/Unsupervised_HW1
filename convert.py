import csv

# Generator function to read the file line by line
def read_large_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

# Function to process each citation entry
def process_entry(lines):
    current_entry = {}
    for line in lines:
        if line.startswith('#*'):
            current_entry['Title'] = line[2:].strip()
        elif line.startswith('#@'):
            current_entry['Authors'] = line[2:].strip()
        elif line.startswith('#t'):
            current_entry['Year'] = line[2:].strip()
        elif line.startswith('#c'):
            current_entry['Venue'] = line[2:].strip()
        elif line.startswith('#index'):
            current_entry['Index'] = line[6:].strip()
        elif line.startswith('#%'):
            references = current_entry.get('References', [])
            references.append(line[2:].strip())
            current_entry['References'] = references
    return current_entry

# Function to handle processing and writing to CSV
def process_large_file(input_path, output_path):
    data = []
    current_lines = []

    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['Title', 'Authors', 'Year', 'Venue', 'Index', 'References']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for line in read_large_file(input_path):
            if line.startswith('#*') and current_lines:
                # Process the current entry and write to CSV
                entry = process_entry(current_lines)
                if 'References' in entry:
                    entry['References'] = ', '.join(entry['References'])
                writer.writerow(entry)
                current_lines = []  # Reset for next entry

            current_lines.append(line)

        # Process the last entry if any
        if current_lines:
            entry = process_entry(current_lines)
            if 'References' in entry:
                entry['References'] = ', '.join(entry['References'])
            writer.writerow(entry)

    print("Processing complete. Check the output CSV file.")

# Call the function with file paths
process_large_file('acm.txt', 'acm.csv')
