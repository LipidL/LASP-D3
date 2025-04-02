for num_atoms in range(5,100000):
    total_interactions = (num_atoms * (num_atoms - 1)) // 2
    print(f"total_interactions: {total_interactions}")
    ij_list = []
    ij_map = {}
    for thread_id in range(0,total_interactions):
        discriminant = (2.0 * num_atoms - 1.0) * (2.0 * num_atoms - 1.0) - 8.0 * thread_id
        i = (2.0 * num_atoms - 1.0 - (discriminant ** 0.5)) // 2
        row_start = i * (2 * num_atoms - 1 - i) // 2
        j = thread_id - row_start + i + 1

        ij_list.append((int(i), int(j)))
        ij_map[(int(i), int(j))] = thread_id

    assert len(ij_list) == total_interactions, f"Expected {total_interactions} interactions, but got {len(ij_list)}"
    assert len(ij_map) == total_interactions, f"Expected {total_interactions} interactions, but got {len(ij_map)}"