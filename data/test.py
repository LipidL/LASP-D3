num_atoms = 5

total_interactions = (num_atoms * (num_atoms - 1)) // 2
print(f"total_interactions: {total_interactions}")
ij_list = []
for thread_id in range(0,total_interactions):
    discriminant = (2.0 * num_atoms - 1.0) * (2.0 * num_atoms - 1.0) - 8.0 * thread_id
    i = (2.0 * num_atoms - 1.0 - (discriminant ** 0.5)) // 2
    row_start = i * (2 * num_atoms - 1 - i) // 2
    j = thread_id - row_start + i + 1

    ij_list.append((int(i), int(j)))

print(f"ij_list: {ij_list}")
print(f"ij_list length: {len(ij_list)}")