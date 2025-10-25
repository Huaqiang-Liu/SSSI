def sum_column(file_path, column_index=0):
    total = 0.0
    with open(file_path, 'r') as file:
        for line in file:
            numbers = line.split()
            if len(numbers) >= 1:
                try:
                    total += float(numbers[column_index])
                except ValueError:
                    print(f"Warning: Could not convert {numbers[0]} to float.")
    return total
if __name__ == "__main__":
    print(sum_column('log_host.txt', 0) + sum_column('log_guest.txt', 0))
