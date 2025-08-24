load("construct_fasta_linear_layer.sage")
import random


class FASTA:
    def __init__(self, b, s, rounds):
        self.b = b
        self.n_keys = s
        self.n_u = b * self.n_keys
        self.var_k = [f"k_{i}" for i in range(self.n_keys)]
        self.var_u = [f"u_{i}_{j}" for i in range(self.b) for j in range(self.n_keys)]

        self.R = BooleanPolynomialRing(self.n_keys + self.n_u, self.var_k + self.var_u)

        self.var_k = list(self.R.gens()[: self.n_keys])
        self.var_u = list(self.R.gens()[self.n_keys :])
        self.var_u = [
            self.var_u[i * self.n_keys : (i + 1) * self.n_keys] for i in range(self.b)
        ]

        self.rounds = rounds

        self.constants = [
            [GF(2).random_element() for _ in range(self.n_u)] for _ in range(rounds + 1)
        ]
        self.matrixs = []

        self.r1_list = []
        self.r2_list = []
        self.r3_base = 5
        self.r3_mod = 3
        self.i_base = [1, 3]
        self.j_base = [2, 4]
        self.l_base = [3, 5]

        if b == 5 and s == 329:
            self.r1_list = [random.choice([1, 2, 3]) for _ in range(4)]
            self.r2_list = [random.choice([4, 5, 6]) for _ in range(4)]
            self.r3_base = 7
            self.i_random = 4
            self.j_random = 18
            self.l_random = 61
        elif b == 3 and s == 5:
            self.r1_list = [random.choice([1, 2]) for _ in range(4)]
            self.r2_list = [random.choice([3, 4]) for _ in range(4)]
            self.r3_base = 5
            self.r3_mod = 3
            self.i_base = [1, 3]
            self.j_base = [2, 4]
            self.l_base = [3, 5]
            self.t = 3
        elif b == 3 and s == 11:
            self.r1_list = [random.choice([1, 2]) for _ in range(4)]
            self.r2_list = [random.choice([3, 4]) for _ in range(4)]
            self.r3_base = 5
            self.r3_mod = 3
            self.i_base = [1, 3]
            self.j_base = [2, 4]
            self.l_base = [3, 5]
            self.t = 3
        elif b == 3 and s == 5:
            self.r1_list = [random.choice([1, 2]) for _ in range(4)]
            self.r2_list = [random.choice([3, 4]) for _ in range(4)]
            self.r3_base = 0
            self.r3_mod = 0
            self.i_base = [1, 2]
            self.j_base = [2, 3]
            self.l_base = [3, 4]
            self.t = 3
        elif b == 3 and s == 7:
            self.r1_list = [random.choice([1, 2]) for _ in range(4)]
            self.r2_list = [random.choice([3, 4]) for _ in range(4)]
            self.r3_base = 5
            self.r3_mod = 3
            self.i_base = [1, 2]
            self.j_base = [2, 3]
            self.l_base = [3, 4]
            self.t = 3
        else:
            self.r1_list = [random.choice([1, 2, 3]) for _ in range(4)]
            self.r2_list = [random.choice([2, 3, 4]) for _ in range(4)]
            self.r3_base = 2

    def chi_operation(self, state):
        output_state = [[0 for _ in range(self.n_keys)] for _ in range(self.b)]
        for i in range(self.b):
            for j in range(self.n_keys):
                output_state[i][j] = (
                    state[i][j]
                    + (state[i][(j + 1) % self.n_keys] + 1)
                    * state[i][(j + 2) % self.n_keys]
                )
        return output_state

    def linear_transformation(self, state, r):
        new_state = [0] * self.n_u
        if r == 0:
            if self.b == 3:
                i = [1, 0]
                j = [0, 1]
                l = [1, 1]
            elif self.b == 5:
                i = [0, 1, 1, 0]
                j = [0, 1, 2, 1]
                l = [1, 2, 0, 1]
            elif self.b == 7:
                i = [1, 1]
                j = [0, 1]
                l = [1, 0]
            if self.n_keys == 5:
                r1_list = [1, 2, 2, 1]
                r2_list = [3, 4, 3, 4]
            elif self.n_keys == 7:
                r1_list = [1, 1, 1, 2]
                r2_list = [3, 3, 4, 3]
            M = build_fasta_linear_layer(
                self.b,
                self.n_keys,
                r1_list,
                r2_list,
                self.r3_base,
                self.r3_mod,
                i,
                j,
                l,
                self.i_base,
                self.j_base,
                self.l_base,
                self.t,
            )
            self.matrixs.append(M)
        else:
            M = Matrix(GF(2), self.n_u, self.n_u, 0)
            while not M.is_invertible():
                i = [random.randint(0, 1) for _ in range(self.b - 1)]
                j = [random.randint(0, 1) for _ in range(self.b - 1)]
                l = [random.randint(0, 1) for _ in range(self.b - 1)]
                self.r1_list = [random.choice([1, 2]) for _ in range(4)]
                self.r2_list = [random.choice([3, 4]) for _ in range(4)]
                M = build_fasta_linear_layer(
                    self.b,
                    self.n_keys,
                    self.r1_list,
                    self.r2_list,
                    self.r3_base,
                    self.r3_mod,
                    i,
                    j,
                    l,
                    self.i_base,
                    self.j_base,
                    self.l_base,
                    self.t,
                )

        self.matrixs.append(M)
        state_1 = []
        for i in state:
            for j in i:
                state_1.append(j)
        # print(state_1)
        new_state = M * vector(state_1)
        if r == 0:
            new_state = list(new_state)
        else:
            new_state = list(new_state + vector(GF(2), self.constants[r]))

        new_state_1 = [
            new_state[i * self.n_keys : (i + 1) * self.n_keys] for i in range(self.b)
        ]
        return new_state_1

    def forward_compute(self):
        self.init_state = [self.var_k]
        for i in range(1, self.b):
            self.init_state.append(self.var_k[i:] + self.var_k[:i])

        forward_state = self.linear_transformation(self.init_state, 0)

        first_linear_state = [[0 for _ in range(self.n_keys)] for _ in range(self.b)]
        # first_linear_state = forward_state
        for i in range(self.b):
            for j in range(self.n_keys):
                first_linear_state[i][j] = forward_state[i][j]
        # print(self.init_state)
        # print(self.matrixs[0])
        # print("first_linear_state:", first_linear_state)
        u_state = [[0 for _ in range(self.n_keys)] for _ in range(self.b)]
        for i in range(self.b):
            for j in range(self.n_keys):
                u_state[i][j] = (
                    first_linear_state[i][(j + 1) % self.n_keys]
                    * first_linear_state[i][(j + 2) % self.n_keys]
                )

        rc_0 = [
            self.constants[0][i * self.n_keys : (i + 1) * self.n_keys]
            for i in range(self.b)
        ]
        # for i in range(self.b):
        #     for j in range(self.n_keys):
        #         forward_state[i][j] = forward_state[i][j] + rc_0[i][j]
        # print(self.first_linear_state)

        for i in range(self.b):
            for j in range(self.n_keys):
                forward_state[i][j] = (
                    forward_state[i][j]
                    + rc_0[i][j]
                    + self.var_u[i][j]
                    + forward_state[i][(j + 1) % self.n_keys]
                    * rc_0[i][(j + 2) % self.n_keys]
                    + forward_state[i][(j + 2) % self.n_keys]
                    + rc_0[i][(j + 2) % self.n_keys]
                    + rc_0[i][(j + 1) % self.n_keys]
                    * forward_state[i][(j + 2) % self.n_keys]
                    + +rc_0[i][(j + 1) % self.n_keys] * rc_0[i][(j + 2) % self.n_keys]
                )

        for r in range(1, self.rounds - 1):
            forward_state = self.linear_transformation(forward_state, r)
            forward_state = self.chi_operation(forward_state)
        forward_state = self.linear_transformation(forward_state, self.rounds - 1)
        # print("first_linear_state:", first_linear_state)
        return forward_state, first_linear_state, u_state

    def backward_compute(self, key_value):
        # m_c = [GF(2).random_element() for _ in range(self.n_u)]
        init_value = []
        for i in range(self.b):
            # print(key_value[i:] + key_value[:i])
            init_value.append(key_value[i:] + key_value[:i])
        state_value = [el for row in init_value for el in row]
        M_last = Matrix(GF(2), self.n_u, self.n_u, 0)
        while not M_last.is_invertible():
            i = [random.randint(0, 1) for _ in range(self.b - 1)]
            j = [random.randint(0, 1) for _ in range(self.b - 1)]
            l = [random.randint(0, 1) for _ in range(self.b - 1)]
            self.r1_list = [random.choice([1, 2]) for _ in range(4)]
            self.r2_list = [random.choice([3, 4]) for _ in range(4)]
            M_last = build_fasta_linear_layer(
                self.b,
                self.n_keys,
                self.r1_list,
                self.r2_list,
                self.r3_base,
                self.r3_mod,
                i,
                j,
                l,
                self.i_base,
                self.j_base,
                self.l_base,
                self.t,
            )
        self.matrixs.append(M_last)
        for r in range(self.rounds + 1):
            # print(state_value, type(state_value))
            state_value = list(
                self.matrixs[r] * vector(GF(2), state_value)
                + vector(GF(2), self.constants[r])
            )
            # print("self.constants[r]):", self.constants[r])
            # print("after linear state value:", state_value, type(state_value))
            state_value = [
                state_value[i * self.n_keys : (i + 1) * self.n_keys] for i in range(b)
            ]
            # print("after linear state value:", state_value)

            if r == self.rounds:
                continue
            for i in range(self.b):
                for j in range(self.n_keys):
                    state_value[i][j] = (
                        state_value[i][j]
                        + (state_value[i][(j + 1) % self.n_keys] + 1)
                        * state_value[i][(j + 2) % self.n_keys]
                    )
            # print("r:", r)
            # print("after chi state value:", state_value)
            state_value = [el for row in state_value for el in row]
        state_value = [el for row in state_value for el in row]
        # print("state_value:", state_value)

        backward_state = []
        # init_state = [el for row in self.init_state for el in row]
        init_state = []
        for i in range(self.b):
            init_state += self.var_k
        # print(self.constants[self.rounds])
        # print(init_state)
        for i in range(self.n_u):
            backward_state.append(
                init_state[i] + state_value[i] + self.constants[self.rounds][i]
            )
        # print(backward_state)
        # i = [random.randint(0, 1) for _ in range(self.b - 1)]
        # j = [random.randint(0, 1) for _ in range(self.b - 1)]
        # l = [random.randint(0, 1) for _ in range(self.b - 1)]
        # M_last = build_fasta_linear_layer(
        #     self.b,
        #     self.n_keys,
        #     self.r1_list,
        #     self.r2_list,
        #     self.r3_base,
        #     i,
        #     j,
        #     l,
        #     self.i_base,
        #     self.j_base,
        #     self.l_base,
        # )
        # print(M_last.det())
        backward_state = list(M_last.inverse() * vector(backward_state))
        backward_state = [
            backward_state[i * self.n_keys : (i + 1) * self.n_keys]
            for i in range(self.b)
        ]
        # print("backward_state:", backward_state)
        return backward_state

    def generate_eqautions_from_last(
        self,
        forward_state,
        backward_state,
        num_equations,
        all_num_monomials,
        num_equations_from_last,
        num_monomials_degree,
    ):
        equations = []
        if self.rounds == 3:
            for i in range(self.b):
                for j in range(self.n_keys):
                    if (
                        num_equations >= all_num_monomials
                        or num_equations_from_last >= num_monomials_degree
                    ):
                        return equations
                    tmp_eq = backward_state[i][(j + 1) % self.n_keys] * (
                        backward_state[i][j] + forward_state[i][j]
                    )
                    equations.append(tmp_eq)
                    # print(num_equations)
                    # print(tmp_eq)
                    num_equations += 1
                    num_equations_from_last += 1

            for i in range(self.b):
                for j in range(self.n_keys):
                    if (
                        num_equations >= all_num_monomials
                        or num_equations_from_last >= num_monomials_degree
                    ):
                        return equations
                    tmp_eq = (
                        backward_state[i][j]
                        + forward_state[i][j]
                        + backward_state[i][(j + 1) % self.n_keys]
                        * (forward_state[i][(j + 2) % self.n_keys] + 1)
                    )
                    equations.append(tmp_eq)
                    # print(num_equations)
                    # print(tmp_eq)
                    num_equations += 1
                    num_equations_from_last += 1
            for i in range(self.b):
                for j in range(self.n_keys):
                    if (
                        num_equations >= all_num_monomials
                        or num_equations_from_last >= num_monomials_degree
                    ):
                        return equations
                    tmp_eq = backward_state[i][(j + 3) % self.n_keys] * (
                        backward_state[i][(j + 2) % self.n_keys]
                        * backward_state[i][(j + 1) % self.n_keys]
                        + backward_state[i][(j + 2) % self.n_keys]
                        + backward_state[i][j]
                        + forward_state[i][j]
                    )
                    equations.append(tmp_eq)
                    # print(num_equations)
                    # print(tmp_eq)
                    num_equations += 1
                    num_equations_from_last += 1
        if self.rounds == 4:
            for i in range(self.b):
                for j in range(self.n_keys):
                    if (
                        num_equations >= all_num_monomials
                        or num_equations_from_last >= num_monomials_degree
                    ):
                        return equations
                    tmp_eq = backward_state[i][(j + 1) % self.n_keys] * (
                        backward_state[i][j] + forward_state[i][j]
                    )
                    equations.append(tmp_eq)
                    # print(num_equations)
                    # print(tmp_eq)
                    num_equations += 1
                    num_equations_from_last += 1

            for i in range(self.b):
                for j in range(self.n_keys):
                    if (
                        num_equations >= all_num_monomials
                        or num_equations_from_last >= num_monomials_degree
                    ):
                        return equations
                    tmp_eq = (
                        backward_state[i][j]
                        + forward_state[i][j]
                        + backward_state[i][(j + 1) % self.n_keys]
                        * (forward_state[i][(j + 2) % self.n_keys] + 1)
                    )
                    equations.append(tmp_eq)
                    # print(num_equations)
                    # print(tmp_eq)
                    num_equations += 1
                    num_equations_from_last += 1
            for i in range(self.b):
                for j in range(self.n_keys):
                    if (
                        num_equations >= all_num_monomials
                        or num_equations_from_last >= num_monomials_degree
                    ):
                        return equations
                    tmp_eq = backward_state[i][(j + 3) % self.n_keys] * (
                        backward_state[i][(j + 2) % self.n_keys]
                        * backward_state[i][(j + 1) % self.n_keys]
                        + backward_state[i][(j + 2) % self.n_keys]
                        + backward_state[i][j]
                        + forward_state[i][j]
                    )
                    equations.append(tmp_eq)
                    # print(num_equations)
                    # print(tmp_eq)
                    num_equations += 1
                    num_equations_from_last += 1
            if self.n_keys > 5:
                for i in range(self.b):
                    for j in range(self.n_keys):
                        if num_equations >= all_num_monomials:
                            return equations
                        equations.append(
                            backward_state[i][(j + 5) % self.n_keys]
                            * (
                                forward_state[i][j]
                                + forward_state[i][(i + 2) % self.n_keys]
                                + backward_state[i][j]
                                + backward_state[i][(j + 1) % self.n_keys]
                                * backward_state[i][(j + 2) % self.n_keys]
                                + backward_state[i][(j + 1) % self.n_keys]
                                * (backward_state[i][(j + 3) % self.n_keys] + 1)
                                * backward_state[i][(j + 4) % self.n_keys]
                            )
                        )
                        num_equations += 1
            if self.n_keys > 7:
                for i in range(self.b):
                    for j in range(self.n_keys):
                        if num_equations >= all_num_monomials:
                            return equations
                        equations.add(
                            backward_state[i][(j + 7) % self.n_keys]
                            * (
                                forward_state[i][j]
                                + backward_state[i][j]
                                + (backward_state[i][(j + 1) % self.n_keys] + 1)
                                * backward_state[i][(j + 2) % self.n_keys]
                                + (backward_state[i][(j + 1) % self.n_keys] + 1)
                                * (backward_state[i][(j + 3) % self.n_keys] + 1)
                                * (
                                    backward_state[i][(j + 4) % self.n_keys]
                                    + (backward_state[i][(j + 5) % self.n_keys] + 1)
                                    * backward_state[i][(j + 6) % self.n_keys]
                                )
                            )
                        )
                        num_equations += 1

        return equations

    def generate_eqautions_from_uf(
        self,
        degree,
        first_linear_state,
        u_state,
        num_equations,
        all_num_monomials,
        num_equations_from_uf,
        num_monomials_degree,
    ):
        equations = []
        # print(degree)
        if degree == 2:
            for i in range(self.b):
                for j in range(self.n_keys):
                    # print(num_equations)
                    if (
                        num_equations >= all_num_monomials
                        or num_equations_from_uf >= num_monomials_degree
                    ):
                        return equations
                    equations.append(self.var_u[i][j] + u_state[i][j])
                    tmp = self.var_u[i][j] + u_state[i][j]
                    # print(num_equations)
                    # print(tmp)
                    # print(tmp.degree())

                    num_equations += 1
                    num_equations_from_uf += 1

            for i in range(self.b):
                for j in range(self.n_keys):
                    # num_equations += 1
                    if (
                        num_equations >= all_num_monomials
                        or num_equations_from_uf >= num_monomials_degree
                    ):
                        return equations
                    tmp_eq = (
                        self.var_u[i][j] * first_linear_state[i][(j + 1) % self.n_keys]
                        + u_state[i][j]
                    )
                    equations.append(tmp_eq)
                    # print(num_equations)
                    # print(tmp_eq)
                    num_equations += 1
                    num_equations_from_uf += 1

            for i in range(self.b):
                for j in range(self.n_keys):
                    # num_equations += 1
                    if (
                        num_equations >= all_num_monomials
                        or num_equations_from_uf >= num_monomials_degree
                    ):
                        return equations
                    tmp_eq = (
                        self.var_u[i][j] * first_linear_state[i][(j + 2) % self.n_keys]
                        + u_state[i][j]
                    )
                    equations.append(tmp_eq)
                    # print(num_equations)
                    # print(tmp_eq)
                    num_equations += 1
                    num_equations_from_uf += 1

            for i in range(self.b):
                for j in range(self.n_keys):
                    # num_equations += 1
                    if (
                        num_equations >= all_num_monomials
                        or num_equations_from_uf >= num_monomials_degree
                    ):
                        return equations
                    tmp_eq = (
                        self.var_u[i][j] * first_linear_state[i][j]
                        + self.var_u[i][(j - 1) % self.n_keys]
                        * first_linear_state[i][(j + 2) % self.n_keys]
                    )
                    equations.append(tmp_eq)
                    # print(num_equations)
                    # print(tmp_eq)
                    num_equations += 1
                    num_equations_from_uf += 1
        if degree == 3:
            # 1. u_i*f_j=f_{i+1}*f_{i+2}*f_j,j!=i+1 and j!=i+2
            for i in range(self.b):
                for j in range(self.n_keys):
                    # print(num_equations)
                    for k in range(self.n_keys):
                        if (
                            num_equations >= all_num_monomials
                            or num_equations_from_uf >= num_monomials_degree
                        ):
                            return equations
                        elif k != (j + 1) % self.n_keys and k != (j + 2) % self.n_keys:
                            # print("k:", k)
                            # print(self.var_u[i][j])
                            # print(first_linear_state[i][(j + 1) % self.n_keys])
                            # print(first_linear_state[i][(j + 2) % self.n_keys])
                            # print(first_linear_state[i][k])
                            tmp = (
                                self.var_u[i][j] * first_linear_state[i][k]
                                + first_linear_state[i][(j + 1) % self.n_keys]
                                * first_linear_state[i][(j + 2) % self.n_keys]
                                * first_linear_state[i][k]
                            )
                            equations.append(tmp)
                            # tmp = self.var_u[i][j] + u_state[i][j]
                            # print(num_equations)
                            # print(tmp)

                            num_equations += 1
                            num_equations_from_uf += 1
            # 2. u_i*u_j=f_{i+1}*f_{i+2}*u_j
            for i in range(self.b):
                for j in range(self.n_keys):
                    # print(num_equations)
                    for k in range(self.n_keys):
                        if (
                            num_equations >= all_num_monomials
                            or num_equations_from_uf >= num_monomials_degree
                        ):
                            return equations
                        # print("k:", k)
                        # print(self.var_u[i][j])
                        # print(first_linear_state[i][(j + 1) % self.n_keys])
                        # print(first_linear_state[i][(j + 2) % self.n_keys])
                        # print(first_linear_state[i][k])
                        tmp = (
                            self.var_u[i][j] * self.var_u[i][k]
                            + first_linear_state[i][(j + 1) % self.n_keys]
                            * first_linear_state[i][(j + 2) % self.n_keys]
                            * self.var_u[i][k]
                        )
                        equations.append(tmp)
                        # tmp = self.var_u[i][j] + u_state[i][j]
                        # print(num_equations)
                        # print(tmp)
                        num_equations += 1
                        num_equations_from_uf += 1
            # 3. u_i*u_{i+1}=f_{i+1}*f_{i+2}*f_{i+3}
            for i in range(self.b):
                for j in range(self.n_keys):
                    # print(num_equations)

                    if (
                        num_equations >= all_num_monomials
                        or num_equations_from_uf >= num_monomials_degree
                    ):
                        return equations
                    tmp = (
                        self.var_u[i][j] * self.var_u[i][(j + 1) % self.n_keys]
                        + first_linear_state[i][(j + 1) % self.n_keys]
                        * first_linear_state[i][(j + 2) % self.n_keys]
                        * first_linear_state[i][(j + 3) % self.n_keys]
                    )
                    equations.append(tmp)
                    # tmp = self.var_u[i][j] + u_state[i][j]
                    # print(num_equations)
                    # print(tmp)
                    num_equations += 1
                    num_equations_from_uf += 1
            # 4. u_i*u_j=f_{i+1}*f_{i+2}*u_j
            for i in range(self.b):
                for j in range(self.n_keys):
                    # print(num_equations)
                    for k in range(self.n_keys):
                        if (
                            num_equations >= all_num_monomials
                            or num_equations_from_uf >= num_monomials_degree
                        ):
                            return equations
                        if k != j:
                            tmp = (
                                self.var_u[i][j] * self.var_u[i][k]
                                + first_linear_state[i][(j + 1) % self.n_keys]
                                * first_linear_state[i][(j + 2) % self.n_keys]
                                * self.var_u[i][k]
                            )
                            equations.append(tmp)
                            # tmp = self.var_u[i][j] + u_state[i][j]
                            # print(num_equations)
                            # print(tmp)
                            num_equations += 1
                            num_equations_from_uf += 1
            # 4. u_i*f_{i+1}*f_j=f_{i+1}*f_{i+2}*f_j,j!=i+1
            for i in range(self.b):
                for j in range(self.n_keys):
                    for k in range(self.n_keys):
                        if (
                            num_equations >= all_num_monomials
                            or num_equations_from_uf >= num_monomials_degree
                        ):
                            return equations
                        if k != (j + 1) % self.n_keys:
                            tmp = (
                                self.var_u[i][j]
                                * first_linear_state[i][k]
                                * first_linear_state[i][(j + 1) % self.n_keys]
                                + first_linear_state[i][k]
                                * first_linear_state[i][(j + 1) % self.n_keys]
                                * first_linear_state[i][(j + 2) % self.n_keys]
                            )
                            equations.append(tmp)
                            # tmp = self.var_u[i][j] + u_state[i][j]
                            # print(num_equations)
                            # print(tmp)
                            num_equations += 1
                            num_equations_from_uf += 1
            # 5. u_i*f_{i+1}*u_j=f_{i+1}*f_{i+2}*u_j
            for i in range(self.b):
                for j in range(self.n_keys):
                    for k in range(self.n_keys):
                        if (
                            num_equations >= all_num_monomials
                            or num_equations_from_uf >= num_monomials_degree
                        ):
                            return equations

                        tmp = (
                            self.var_u[i][j]
                            * self.var_u[i][k]
                            * first_linear_state[i][(j + 1) % self.n_keys]
                            + self.var_u[i][k]
                            * first_linear_state[i][(j + 1) % self.n_keys]
                            * first_linear_state[i][(j + 2) % self.n_keys]
                        )
                        equations.append(tmp)
                        # tmp = self.var_u[i][j] + u_state[i][j]
                        # print(num_equations)
                        # print(tmp)
                        num_equations += 1
                        num_equations_from_uf += 1
            # 6. u_i*f_{i+2}*f_j = f_{i+1}*f_{i+2}*f_j, j!=i+1,i+2
            for i in range(self.b):
                for j in range(self.n_keys):
                    for k in range(self.n_keys):
                        if (
                            num_equations >= all_num_monomials
                            or num_equations_from_uf >= num_monomials_degree
                        ):
                            return equations
                        if k != (j + 1) % self.n_keys and k != (j + 2) % self.n_keys:
                            tmp = (
                                self.var_u[i][j]
                                * first_linear_state[i][(j + 2) % self.n_keys]
                                * first_linear_state[i][k]
                                + first_linear_state[i][(j + 1) % self.n_keys]
                                * first_linear_state[i][(j + 2) % self.n_keys]
                                * first_linear_state[i][k]
                            )
                            equations.append(tmp)
                            # tmp = self.var_u[i][j] + u_state[i][j]
                            # print(num_equations)
                            # print(tmp)
                            num_equations += 1
                            num_equations_from_uf += 1
            # 7. u_i*f_{i+2}*u_j = f_{i+1}*f_{i+2}*u_j
            for i in range(self.b):
                for j in range(self.n_keys):
                    for k in range(self.n_keys):
                        if (
                            num_equations >= all_num_monomials
                            or num_equations_from_uf >= num_monomials_degree
                        ):
                            return equations

                        tmp = (
                            self.var_u[i][j]
                            * first_linear_state[i][(j + 2) % self.n_keys]
                            * self.var_u[i][k]
                            + first_linear_state[i][(j + 1) % self.n_keys]
                            * first_linear_state[i][(j + 2) % self.n_keys]
                            * self.var_u[i][k]
                        )
                        equations.append(tmp)
                        # tmp = self.var_u[i][j] + u_state[i][j]
                        # print(num_equations)
                        # print(tmp)
                        num_equations += 1
                        num_equations_from_uf += 1
            # 8. u_i*f_i*f_j = f_{i+2}*f_j*u_{i-1}
            for i in range(self.b):
                for j in range(self.n_keys):
                    for k in range(self.n_keys):
                        if (
                            num_equations >= all_num_monomials
                            or num_equations_from_uf >= num_monomials_degree
                        ):
                            return equations

                        tmp = (
                            self.var_u[i][j]
                            * first_linear_state[i][j]
                            * first_linear_state[i][k]
                            + first_linear_state[i][(j + 2) % self.n_keys]
                            * first_linear_state[i][k]
                            * self.var_u[i][(j - 1) % self.n_keys]
                        )
                        equations.append(tmp)
                        # tmp = self.var_u[i][j] + u_state[i][j]
                        # print(num_equations)
                        # print(tmp)
                        num_equations += 1
                        num_equations_from_uf += 1
            # 9. u_i*f_i*u_j = f_{i+2}*u_j*u_{i-1}
            for i in range(self.b):
                for j in range(self.n_keys):
                    for k in range(self.n_keys):
                        if (
                            num_equations >= all_num_monomials
                            or num_equations_from_uf >= num_monomials_degree
                        ):
                            return equations

                        tmp = (
                            self.var_u[i][j]
                            * first_linear_state[i][j]
                            * self.var_u[i][k]
                            + first_linear_state[i][(j + 2) % self.n_keys]
                            * self.var_u[i][k]
                            * self.var_u[i][(j - 1) % self.n_keys]
                        )
                        equations.append(tmp)
                        # tmp = self.var_u[i][j] + u_state[i][j]
                        # print(num_equations)
                        # print(tmp)
                        num_equations += 1
                        num_equations_from_uf += 1
        return equations

    def generate_boolean_monomials(self, degree, first_linear_state):

        new_sirst_linear_state = [
            element for row in first_linear_state for element in row
        ]
        # print(new_sirst_linear_state)

        vars = [
            element for row in self.var_u for element in row
        ] + new_sirst_linear_state
        # vars = [element for row in self.var_u for element in row] + self.var_k
        # print(vars)

        from itertools import combinations

        monomials = [self.R(1)]
        for deg in range(1, degree + 1):
            for subset in combinations(vars, deg):
                # print(subset)
                monomial = prod(subset)
                if monomial != self.R(1):
                    monomials.append(monomial)
        return monomials

    def generate_eqautions_from_all(
        self, degree, first_linear_state, u_state, num_equations, all_num_monomials
    ):
        degree_monomials = self.generate_boolean_monomials(degree, first_linear_state)
        equations = []
        # print(degree_monomials)
        # print(len(degree_monomials))
        for monomial in degree_monomials:
            for i in range(self.b):
                for j in range(self.n_keys):
                    if monomial != self.R(1):
                        # print(monomial)
                        tmp_s = (self.var_u[i][j] + u_state[i][j]) * monomial
                        if tmp_s.degree() == degree and tmp_s != self.R(0):
                            if num_equations >= all_num_monomials:
                                return equations
                            # print(self.var_u[i][j] + u_state[i][j])
                            # print(monomial)
                            equations.append(tmp_s)
                            # print(num_equations)
                            # print("tmp_s:", tmp_s)
                            # print()
                            # print(self.var_u[i][j] + u_state[i][j])
                            num_equations += 1
        return equations


result = {}


def gaussian_elimination_bool(matrix):
    m = len(matrix)
    if m == 0:
        return [], [], [], {}
    n = len(matrix[0])

    basis = []

    pivot_cols = []

    v_basis = []

    dependencies = {}

    independent_rows = []

    mat = [row[:] for row in matrix]

    for i in range(m):

        v = set([i])

        row = mat[i][:]

        for j in range(len(basis)):
            p = pivot_cols[j]
            if row[p] == 1:

                for k in range(n):
                    row[k] = (row[k] + basis[j][k]) % 2

                v = v.symmetric_difference(v_basis[j])

        if all(x == 0 for x in row):

            v.discard(i)
            dependencies[i] = v
        else:

            p = 0
            while p < n and row[p] == 0:
                p += 1
            if p < n:  # 找到主元列
                pivot_cols.append(p)
                basis.append(row)
                v_basis.append(v)
                independent_rows.append(i)

    return independent_rows, basis, pivot_cols, dependencies


import numpy as np

# import numba


# @numba.jit(nopython=True, parallel=True) #parallel speeds up computation only over very large matrices
# M is a mxn matrix binary matrix
# all elements in M should be uint8
def gf2elim(M):

    m, n = M.shape

    i = 0
    j = 0

    while i < m and j < n:
        # find value and index of largest element in remainder of column j
        k = np.argmax(M[i:, j]) + i

        # swap rows
        # M[[k, i]] = M[[i, k]] this doesn't work with numba
        temp = np.copy(M[k])
        M[k] = M[i]
        M[i] = temp

        aijn = M[i, j:]

        col = np.copy(M[:, j])  # make a copy otherwise M will be directly affected

        col[i] = 0  # avoid xoring pivot row with itself

        flip = np.outer(col, aijn)

        M[:, j:] = M[:, j:] ^ flip

        i += 1
        j += 1

    return M


import sys

# Accept parameters b, s, rounds from the command line input (arguments passed when running the .sage script)
# b = int(sys.argv[1])  # b parameter
# s = int(sys.argv[2])  # s parameter
# rounds = int(sys.argv[3])  # rounds parameter

for i in range(100):
    b = int(sys.argv[1])  # b parameter
    s = int(sys.argv[2])  # s parameter
    rounds = int(sys.argv[3])  # rounds parameter
    all_num_monomials = 0
    for i in range(2 ^ (rounds - 2) + 2):
        all_num_monomials += binomial(b * s + s, i)
    all_num_monomials = all_num_monomials - binomial(b * s, 2 ^ (rounds - 2) + 1) - 1

    num_monomials_degree_1 = binomial(b * s + s, 1)
    num_monomials_degree_2 = (
        binomial(b * s + s, 2) + binomial(b * s + s, 1) - binomial(b * s, 2)
    )
    print("num_monomials_degree_2:", num_monomials_degree_2)
    num_monomials_degree_3 = (
        binomial(b * s + s, 3)
        + binomial(b * s + s, 2)
        + binomial(b * s + s, 1)
        - binomial(b * s, 3)
    )
    print("num_monomials_degree_3:", num_monomials_degree_3)
    num_monomials_degree_4 = (
        binomial(b * s + s, 4)
        + binomial(b * s + s, 3)
        + binomial(b * s + s, 2)
        + binomial(b * s + s, 1)
        - binomial(b * s, 4)
    )
    print("num_monomials_degree_4:", num_monomials_degree_4)
    num_monomials_degree_5 = all_num_monomials
    print("num_monomials_degree_5:", num_monomials_degree_5)

    # all_num_monomials = (
    #     num_monomials_degree_4 + num_monomials_degree_3 + num_monomials_degree_2
    # )
    all_num_monomials = num_monomials_degree_3
    print("all_num_monomials:", all_num_monomials)
    # all_num_monomials = 280
    # data_complex_1 = 0
    # data_complex_2 = 0
    # if rounds == 3:
    #     data_complex_1 = ceil(all_num_monomials / (3 * b * s))
    #     data_complex_2 = ceil()
    # elif rounds == 4:
    #     if s > 5:
    #         data_complex = ceil(all_num_monomials / (4 * b * s))
    #     if s > 7:
    #         data_complex = ceil(all_num_monomials / (5 * b * s))
    #     else:
    #         data_complex = ceil(all_num_monomials / (3 * b * s))

    # print("all_num_monomials:", all_num_monomials)
    # print("data_complex:", data_complex)
    key_value = [GF(2).random_element() for _ in range(s)]
    print("key value:", key_value)
    equations_from_last = []
    equations_from_uf_degree_2 = []
    equations_from_uf_degree_3 = []
    equations_from_uf_degree_4 = []
    equations_from_all = []
    # equations_new = []
    monomial_set = set()
    # all_num_monomials = 1760
    # for _ in range(data_complex + 1):
    while True:
        fasta = FASTA(b, s, rounds)
        # print(fasta.constants)
        forward_state, first_linear_state, u_state = fasta.forward_compute()
        # print(fasta.matrixs)
        # print("u_state:", u_state)
        # print("forward_state:", forward_state)
        # print("first_linear_state:", first_linear_state)
        backward_state = fasta.backward_compute(key_value)
        # print(backward_state)

        num_equations_from_last = len(equations_from_last)
        num_equations_from_uf_degree_2 = len(equations_from_uf_degree_2)
        num_equations_from_uf_degree_3 = len(equations_from_uf_degree_3)
        num_equations_from_all = len(equations_from_all)
        num_equations = (
            num_equations_from_last
            + num_equations_from_uf_degree_2
            + num_equations_from_uf_degree_3
            + num_equations_from_all
        )
        if num_equations >= all_num_monomials:
            break

        new_equations_1 = fasta.generate_eqautions_from_last(
            forward_state,
            backward_state,
            num_equations,
            all_num_monomials,
            num_equations_from_last,
            num_monomials_degree_5,
        )
        for eq in new_equations_1:
            monomial_set = union(monomial_set, eq.monomials())
            # print(eq.degree())
        equations_from_last += new_equations_1
        num_equations_from_last = len(equations_from_last)
        print("first monomial_set:", len(monomial_set))
        print("first equations:", num_equations_from_last)

        # num_equations_from_last = len(equations_from_last)
        # num_equations_from_uf_degree_2 = len(equations_from_uf_degree_2)
        # num_equations_from_uf_degree_3 = len(equations_from_uf_degree_3)
        # num_equations_from_all = len(equations_from_all)
        # num_equations = (
        #     num_equations_from_last
        #     + num_equations_from_uf_degree_2
        #     + num_equations_from_uf_degree_3
        #     + num_equations_from_all
        # )
        # if num_equations >= all_num_monomials:
        #     break
        # if num_equations_from_uf_degree_2 < num_monomials_degree_2:

        #     new_equations_2 = fasta.generate_eqautions_from_uf(
        #         2,
        #         first_linear_state,
        #         u_state,
        #         num_equations,
        #         all_num_monomials,
        #         num_equations_from_uf_degree_2,
        #         num_monomials_degree_2,
        #     )
        #     for eq in new_equations_2:
        #         # print(eq)
        #         monomial_set = union(monomial_set, eq.monomials())
        #         # print(eq.monomials())
        #         # print(monomial_set)
        #     equations_from_uf_degree_2 += new_equations_2
        #     print(
        #         "after generate_eqautions_from_uf degree 2 monomial_set:",
        #         len(monomial_set),
        #     )
        #     # print(num_equations)
        #     num_equations_from_uf_degree_2 = len(equations_from_uf_degree_2)
        #     print("num_equations_from_uf_degree_2:", num_equations_from_uf_degree_2)

        # num_equations_from_last = len(equations_from_last)
        # num_equations_from_uf_degree_2 = len(equations_from_uf_degree_2)
        # num_equations_from_uf_degree_3 = len(equations_from_uf_degree_3)
        # num_equations_from_all = len(equations_from_all)
        # num_equations = (
        #     num_equations_from_last
        #     + num_equations_from_uf_degree_2
        #     + num_equations_from_uf_degree_3
        #     + num_equations_from_all
        # )
        # if num_equations >= all_num_monomials:
        #     break
        # if num_equations_from_uf_degree_3 < num_monomials_degree_3:

        #     new_equations_2 = fasta.generate_eqautions_from_uf(
        #         3,
        #         first_linear_state,
        #         u_state,
        #         num_equations,
        #         all_num_monomials,
        #         num_equations_from_uf_degree_3,
        #         num_monomials_degree_3,
        #     )
        #     for eq in new_equations_2:
        #         # print(eq)
        #         monomial_set = union(monomial_set, eq.monomials())
        #         # print(eq.monomials())
        #         # print(monomial_set)
        #     equations_from_uf_degree_3 += new_equations_2
        #     print(
        #         "after generate_eqautions_from_uf degree 3 monomial_set:",
        #         len(monomial_set),
        #     )
        #     # print(num_equations)
        #     num_equations_from_uf_degree_3 = len(equations_from_uf_degree_3)
        #     print("num_equations_from_uf_degree_3:", num_equations_from_uf_degree_3)

        # num_equations_from_last = len(equations_from_last)
        # num_equations_from_uf_degree_2 = len(equations_from_uf_degree_2)
        # num_equations_from_uf_degree_3 = len(equations_from_uf_degree_3)
        # num_equations_from_all = len(equations_from_all)
        # num_equations = (
        #     num_equations_from_last
        #     + num_equations_from_uf_degree_2
        #     + num_equations_from_uf_degree_3
        #     + num_equations_from_all
        # )
        # if num_equations >= all_num_monomials:
        #     break
        # # if num_equations_from_uf >= num_monomials_degree_2:
        # #     continue

        # new_equations_3 = fasta.generate_eqautions_from_all(
        #     4, first_linear_state, u_state, num_equations, all_num_monomials
        # )
        # for eq in new_equations_3:
        #     # print(eq)
        #     # print(eq.monomials())
        #     monomial_set = union(monomial_set, eq.monomials())
        # # print()
        # equations_from_all += new_equations_3
        # # equations_from_all.update(new_equations_3)
        # print("third monomial_set:", len(monomial_set))
        # print("third equations:", num_equations_from_all)

        # print(monomial_set)
        # if num_equations >= all_num_monomials:
        #     break
    # print(equations_new[0] == equations_new[1])
    equations = (
        equations_from_last
        + equations_from_uf_degree_2
        + equations_from_uf_degree_3
        + equations_from_all
    )
    print("Num of equations:", len(equations))
    print("Num of monomials", len(monomial_set))
    print("from equations_from_uf_degree_2:", num_equations_from_uf_degree_2)
    print("from equations_from_uf_degree_3:", num_equations_from_uf_degree_3)
    print("from equations_from_last:", num_equations_from_last)
    print("from equations_from_all:", num_equations_from_all)
    # print(monomial_set)
    # # print(Ms)
    # new_monomial_set = set()
    # for s in Ms:
    #     new_monomial_set.update(s.monomials())
    # print(len(monomial_set))
    print("Constructing coefficient matrix...")
    if fasta.R(1) in monomial_set:
        monomial_set.remove(fasta.R(1))
    coefficient_matrix = zero_matrix(GF(2), len(equations), len(monomial_set))
    var_k = list(fasta.R.gens()[:s])
    var_u = list(fasta.R.gens()[s:])
    for i, eq in enumerate(equations):
        for monomial in eq.monomials():
            # if
            # degree_u = 0
            # degree_k = 0
            # for var in var_u:
            #     degree_u += monomial.degree(var)
            # for var in var_k:
            #     degree_k += monomial.degree(var)
            # if degree_u * 2 + degree_k <= 2 ^ (rounds - 2) + 1:
            if monomial in monomial_set:
                j = list(monomial_set).index(monomial)
                coefficient_matrix[i, j] = 1

    # ind_rows, result_mat, pivots, deps = gaussian_elimination_bool(
    #     list(coefficient_matrix)
    # )
    # print("Linear independent rows:", ind_rows)
    # # print("Row echelon form matrix:")
    # # for r in result_mat:
    # #     print(" ", r)
    # print("Pivot column indices:", pivots)
    # print("Rank:", len(pivots))
    # print("Dependencies of linearly dependent rows:")
    # for dep_row, basis_set in deps.items():
    #     print(f"  Row {dep_row} is XORed from rows {list(basis_set)}")

    print("Calculating rank...")
    rk = coefficient_matrix.rank()
    print("Rank:", rk)
    depend = len(equations) - rk
    if result.get(depend):
        result[depend] += 1
    else:
        result[depend] = 1

print(result)
