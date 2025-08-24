def build_fasta_linear_layer(b, s, r1, r2, r3_base, r3_mod, i, j, l, ii, jj, ll, T):
    # Basic parameters
    # b = 5                   # Number of words
    # s = 19                  # Number of bits per word
    F = GF(2)  # Galois field F₂
    I_s = identity_matrix(F, s)
    I_bs = identity_matrix(F, b * s)

    # Left shift by r positions cyclic shift matrix
    def rot_matrix(s, r):
        return matrix(
            F, s, [(1 if j == (i + r) % s else 0) for i in range(s) for j in range(s)]
        )

    # Theta matrix: r₃ = 7 + ((2r₁ + r₂ + 1) mod 3)
    def theta_matrix(r1, r2):
        r3 = r3_base + ((2 * r1 + r2 + 1) % r3_mod)
        return I_s + rot_matrix(s, r1) + rot_matrix(s, r2) + rot_matrix(s, r3)

    # Construct column parity mixer
    def column_parity_mixer(theta):
        B = block_matrix(F, [[theta] for _ in range(b)])  # size: bs × s
        C = block_matrix(F, [[I_s] * b])  # size: s × bs
        return I_bs + B * C

    # Construct rotation matrix
    def rotation_matrix(offsets):  # offsets = [r0, r1, r2, r3, r4]
        blocks = [rot_matrix(s, r) for r in offsets]
        return block_diagonal_matrix(blocks)

    # Set rotation offsets for each round (w₀ stays fixed)
    R1_offsets = [0]
    R2_offsets = [0]
    R3_offsets = [0]
    for r in range(b - 1):
        R1_offsets.append(ii[r] + i[r])
        R2_offsets.append(jj[r] + j[r])
        R3_offsets.append(ll[r] + l[r])

    # Theta parameters for each round (fixed design, adjustable)
    theta1 = theta_matrix(r1[0], r2[0])
    theta2 = theta_matrix(r1[1], r2[1])
    theta3 = theta_matrix(r1[2], r2[2])
    theta4 = theta_matrix(r1[3], r2[3])

    # Construct P and R for each layer
    P1 = column_parity_mixer(theta1)
    R1 = rotation_matrix(R1_offsets)
    P2 = column_parity_mixer(theta2)
    R2 = rotation_matrix(R2_offsets)
    P3 = column_parity_mixer(theta3)
    R3 = rotation_matrix(R3_offsets)
    P4 = column_parity_mixer(theta4)

    # Permutation = [P1, P2, P3, P4]
    # Rotate = [R1, R2, R3]

    if T == 2:
        M = P2 * R1 * P1
    if T == 3:
        M = P3 * R2 * P2 * R1 * P1
    if T == 4:
        # Final linear layer matrix
        M = P4 * R3 * P3 * R2 * P2 * R1 * P1
    return M


# Example input parameters
# r1 = [2,1,2,2]
# r2 = [4,4,6,4]
# i = [0, 1, 1, 0]
# j = [0, 1, 2, 1]
# l = [1, 2, 1, 0]
# import random

# # Randomly select r1, r2 ∈ [1, 3], [4, 6]
# r1_list = [random.choice([1, 2, 3]) for _ in range(4)]
# r2_list = [random.choice([4, 5, 6]) for _ in range(4)]
# # Randomly select i, j, l ∈ [0, 2]
# i = [random.randint(0, 1) for _ in range(4)]
# j = [random.randint(0, 2) for _ in range(4)]
# l = [random.randint(0, 2) for _ in range(4)]

# # Call the function
# M = build_fasta_linear_layer(5, 19, r1_list, r2_list, i, j, l)
# print("Linear layer dimensions:", M.nrows(), "×", M.ncols())
# print(block_matrix(M))
