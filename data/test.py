from math import sqrt


for num_atoms in range(5,100000):
    total_interactions = (num_atoms * (num_atoms - 1)) // 2
    print(f"total_interactions: {total_interactions}")
    ij_list = []
    ij_map = {}
    bias = 1e-4
    for thread_id in range(0,total_interactions):
        i = (1 + sqrt(1 + 8 * thread_id)) // 2
        j = thread_id - (i * (i - 1)) // 2

        ij_list.append((int(i), int(j)))
        ij_map[(int(i), int(j))] = thread_id
        assert(i != j), f"i and j should not be equal: {i} == {j}"
        assert(i < num_atoms), f"i should be less than num_atoms: {i} >= {num_atoms}"
        assert(j < num_atoms), f"j should be less than num_atoms: {j} >= {num_atoms}"
        assert(i*(i-1)//2 + j == thread_id), f"i*(i-1)//2 + j should be equal to thread_id: {i*(i-1)//2 + j} != {thread_id}"
        assert(i*(i-1)//2 <= thread_id), f"i*(i-1)//2 should be less than or equal to thread_id: {i*(i-1)//2} > {thread_id}"

    assert len(ij_list) == total_interactions, f"Expected {total_interactions} interactions, but got {len(ij_list)}"
    assert len(ij_map) == total_interactions, f"Expected {total_interactions} interactions, but got {len(ij_map)}"